import torchvision.transforms as transforms
from datasets import RegistrationDataset
from model import RegistrationNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from glob import glob
import re
from optparse import OptionParser

def get_args():
    parser = OptionParser()
    parser.add_option('--datadir', "--dd",
                      help='Directory contains 2D images.')
    parser.add_option("-g", '--gpu_id', dest="gpu_id", type='int',
                      help='GPU number, default is None (-g 0 means use gpu 0)')
    parser.add_option('--logdir', "--ld",
                      help='Directory for saving logs and checkpoints')
    parser.add_option('--learning_rate', '--lr', default=1e-4, type='float',
                      help='learning rate for the model')
    parser.add_option('--batchsize', '--bs', dest='batchsize',
                      default=16, type='int', help='batch size for training')
    parser.add_option('-e', '--epochs', default=100, type='int',
                      help='Number of epochs to train')
    parser.add_option('-s', '--shape', default="perspective", type='string',
                      help='Shape transform to use (perspective, affine)')
    parser.add_option('-l', '--loss', default="l2", type='string',
                      help='Loss to use (l2, ssd, ncc)')
    parser.add_option('-a', '--alpha', default="100", type='int',
                      help='Alpha value to use for variational loss')

    (options, args) = parser.parse_args()
    return options


def normalize1(x):
    return 2 * (x - torch.min(x))/(torch.max(x) - torch.min(x)) - 1

class Trainer:

    def __init__(self):

        self.args = get_args()
        os.environ["CUDA_VISIBLE_DEVICES"]= f"{self.args.gpu_id}"
        self.device = torch.device(f"cuda:0")
        print("Using device:", self.device)

        self.checkpoint_dir = os.path.join(f"{self.args.logdir}", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.summary_train = SummaryWriter(os.path.join(self.args.logdir, "train"))
        # self.summary_eval = SummaryWriter(os.path.join(log_dir, "eval"))
    
        #Set Up Dataset and Dataloader
        self.get_transforms()
        self.trainset =  RegistrationDataset(os.path.join(f"{self.args.datadir}", "train"), self.transform, self.shape_transform, self.contrast_transform)
        self.testset = RegistrationDataset(os.path.join(f"{self.args.datadir}", "test"), self.transform, self.shape_transform, self.contrast_transform)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, self.args.batchsize, shuffle=True, num_workers=2)
        self.trainloader = torch.utils.data.DataLoader(self.testset, self.args.batchsize, shuffle=False, num_workers=2)
        
        self.net = RegistrationNet().to(self.device)
        self.criterion = self.variational_loss
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.learning_rate) 

        self.restore_model()

    def get_transforms(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((256, 256))])

        self.contrast_transform = transforms.Compose(
            [transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.4),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2)])

        if self.args.shape == "perspective":
            self.shape_transform = transforms.Compose([transforms.RandomPerspective(distortion_scale=0.6, p=1.0)])
        else:
            self.shape_transform = transforms.Compose([transforms.RandomAffine(degrees = 30, translate=None)])
            
    def variational_loss(self, resampled, fixed, deform_field):
        if self.args.loss == "l2":
            l2 = nn.MSELoss()
            loss = l2(resampled,fixed)
        elif self.args.loss == "ssd":
            loss = torch.mean((resampled-fixed)**2) 
        else: #loss = ncc
            loss = torch.dot(resampled/torch.norm(resampled), fixed/torch.norm(fixed))  #tru mean instead of norm

        alpha = self.args.alpha #TODO mess around with alpha and regularizer values
        w_variance = torch.mean(torch.pow(deform_field[:,:,:,:-1] - deform_field[:,:,:,1:], 2))
        h_variance = torch.mean(torch.pow(deform_field[:,:,:-1,:] - deform_field[:,:,1:,:], 2))
        
        # if torch.any(torch.isnan(torch.pow(deform_field[:,:,:,:-1] - deform_field[:,:,:,1:], 2))) or torch.any(torch.isnan(torch.pow(deform_field[:,:,:-1,:] - deform_field[:,:,1:,:], 2))):    
        print("\n\n")
        print("LOSS INFO")
        print(f"l2: {loss}")
        print(f"deform field: {torch.any(torch.isnan(deform_field))}")
        print(f"w subtraction: {torch.any(torch.isnan(deform_field[:,:,:,:-1] - deform_field[:,:,:,1:]))}")
        print(f"h subtraction: {torch.any(torch.isnan(deform_field[:,:,:-1,:] - deform_field[:,:,1:,:]))}")
        print(f"w_variance: {w_variance}")
        print(f"h_variance: {h_variance}")
        print("\n\n")

        variational = alpha *(h_variance + w_variance)

        #regularizer = nn.L1Loss()(deform_field, torch.zeros_like(deform_field))

        return loss + variational, loss, variational # + 0.1 * regularizer
    
    def restore_model(self, checkpoint_path=""):
            """Restore model checkpoint (if any) and continue training from there.

            Parameters
            ----------
            checkpoint_path : str
                (Optional) If specified, the path for the checkpoint to restore. Otherwise, load the latest checkpoint in
                the log directory.
            """

            if not checkpoint_path:
                checkpoint_path = sorted(glob(os.path.join(self.checkpoint_dir, "*")),
                                        key=lambda x: int(re.match(".*[a-z]+(\d+).pth", x).group(1)))
                if checkpoint_path:
                    checkpoint_path = checkpoint_path[-1]

            if checkpoint_path:
                print(f"Loading model from: {checkpoint_path}")

                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.net.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
                self.start_epoch = checkpoint["epoch"]
                print(f"Finish restoring model. Checkpoint epoch: {self.start_epoch}")

            else:
                print("No saved model found. Training from scratch.")
                self.start_epoch = 0

    def save_model(self, epoch, filepath="", **kwargs):
            """Save a model checkpoint.
            Parameters
            ----------
            epoch : int
                The current epoch number.
            filepath : str
                Optional. Filepath for the checkpoint. If not specified will store at checkpoints directory with epoch
                number in name.
            **kwargs
                Extra parameters to save in the checkpoint.
            """

            # TODO: Only keep last N checkpoints

            if not filepath:
                filepath = os.path.join(self.checkpoint_dir, f'ckpt{epoch + 1}.pth')

            torch.save({
                "epoch": epoch + 1,
                "state_dict": self.net.state_dict(),
                "optimizer_dict":self.optimizer.state_dict(),
                **kwargs
            }, filepath)

    def train(self):
        nan = False

        #Train Model
        for epoch in tqdm(range(self.start_epoch, self.args.epochs), "epochs"):  # loop over the dataset multiple times
            running_loss = []
            running_l2 = []
            running_variational = []
            for i, data in tqdm(enumerate(self.trainloader, 0), "steps", total=len(self.trainloader)):
                # get the inputs; data is a list of [inputs, labels]
                moving, fixed, fixed_contrast = data
                moving, fixed, fixed_contrast = moving.to(self.device), fixed.to(self.device), fixed_contrast.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                deform_fields_small = self.net(moving.detach(), fixed.detach())

                if torch.any(torch.isnan(deform_fields_small)):
                    print("Output is Nan")
                    nan=True
                    break
                
                deform_fields = nn.UpsamplingBilinear2d(scale_factor=4)(deform_fields_small)
                x, y = torch.meshgrid(torch.linspace(-1, 1, 256).to(self.device),
                                    torch.linspace(-1, 1, 256).to(self.device))

                grid_output_x = deform_fields[:, 0, :, :]
                grid_output_y = deform_fields[:, 1, :, :]

                x_resample = x[None] + grid_output_x 
                y_resample = y[None] + grid_output_y

                resamp_grid = torch.stack((y_resample, x_resample), dim=-1).float()
                
                resampled = F.grid_sample(moving, resamp_grid) #apply deformation field to input
                loss, l2_loss, variational = self.criterion(resampled, fixed_contrast, deform_fields_small)
                if torch.any(torch.isnan(loss)):
                    print("Loss is Nan")
                    nan=True
                    break
                loss.backward()
                self.optimizer.step()

                running_loss.append(loss.item())
                running_l2.append(l2_loss.item())
                running_variational.append(variational.item())
            
            if nan:
                break

            # print statistics
            max_to_plot = 5
            self.summary_train.add_images("moving", torch.clip(moving[:max_to_plot], 0, 1), epoch) 
            self.summary_train.add_images("resampled", torch.clip(resampled[:max_to_plot], 0, 1), epoch)
            self.summary_train.add_images("fixed", torch.clip(fixed[:max_to_plot], 0, 1), epoch)
            self.summary_train.add_images("fixed contrast", torch.clip(fixed_contrast[:max_to_plot], 0, 1), epoch)
            
            deform_fields = (deform_fields + 2) / 4
            self.summary_train.add_images("deformation field x", torch.clip(deform_fields[:max_to_plot, 0:1], 0, 1), epoch)
            self.summary_train.add_images("deformation field y", torch.clip(deform_fields[:max_to_plot, 1:2], 0, 1), epoch)
            
            self.summary_train.add_scalar("Loss", np.mean(running_loss), epoch)
            self.summary_train.add_scalar("L2 Loss", np.mean(running_l2), epoch)
            self.summary_train.add_scalar("Variational Loss", np.mean(running_variational), epoch)
            
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, np.mean(running_loss)))
            if epoch % 3 == 0:
                self.save_model(epoch)
        print('Finished Training') 

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
