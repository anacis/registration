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



def normalize1(x):
    return 2 * (x - torch.min(x))/(torch.max(x) - torch.min(x)) - 1


def restore_model(self, checkpoint_path=""):
        """Restore model checkpoint (if any) and continue training from there.

        Parameters
        ----------
        checkpoint_path : str
            (Optional) If specified, the path for the checkpoint to restore. Otherwise, load the latest checkpoint in
            the log directory.
        """

        if not checkpoint_path:
            checkpoint_path = sorted(glob(os.path.join(self.checkpoint_directory, "*")),
                                     key=lambda x: int(re.match(".*[a-z]+(\d+).pth", x).group(1)))
            if checkpoint_path:
                checkpoint_path = checkpoint_path[-1]

        if checkpoint_path:
            print(f"Loading model from: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
            self.start_epoch = checkpoint["epoch"]


            print(f"Finish restoring model. Checkpoint epoch: {self.start_epoch}")

        else:
            print("No saved model found. Training from scratch.")

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
            filepath = os.path.join(self.checkpoint_directory, f'ckpt{epoch + 1}.pth')

        torch.save({
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer_dict": self.optimizer.state_dict(),
            **kwargs
        }, filepath)


def main():
    #Loading Data and Setting Up Transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.RandomCrop((256, 256))])

    target_transform = transforms.Compose(
        [transforms.RandomAffine(degrees = 30, translate=None)])
        #    [transforms.RandomPerspective(distortion_scale=0.6, p=1.0)])

    batch_size = 4

    trainset = RegistrationDataset("/mikQNAP/NYU_knee_data/knee_train_h5/data/train", transform, target_transform)

    testset = RegistrationDataset("/mikQNAP/NYU_knee_data/knee_train_h5/data/test", transform, target_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    net = RegistrationNet()



    #Set up Tensorboard
    log_dir = "../checkpoints/heartvista_simple/logs_rotation"
    summary_train = SummaryWriter(os.path.join(log_dir, "train"))
    summary_eval = SummaryWriter(os.path.join(log_dir, "eval"))

    #Set up loss and optimizer
    def variational_loss(resampled, fixed, deform_field):
        l2 = nn.MSELoss()
        l2_loss = l2(resampled,fixed)
    
        alpha = 10 #TODOL mess around with alpha and regularizer values
        w_variance = torch.mean(torch.pow(deform_field[:,:,:,:-1] - deform_field[:,:,:,1:], 2))
        h_variance = torch.mean(torch.pow(deform_field[:,:,:-1,:] - deform_field[:,:,1:,:], 2))
        variational = alpha *(h_variance + w_variance)
    
        regularizer = nn.L1Loss()(deform_field, torch.zeros_like(deform_field))
    
        return l2_loss + variational, l2_loss, variational # + 0.1 * regularizer


    criterion = variational_loss
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    #Train Model
    for epoch in tqdm(range(100), "epochs"):  # loop over the dataset multiple times

        running_loss = []
        running_l2 = []
        running_variational = []
        for i, data in tqdm(enumerate(trainloader, 0), "steps", total=len(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            moving, fixed = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            deform_fields_small = net(moving, fixed)

            deform_fields = nn.UpsamplingBilinear2d(scale_factor=4)(deform_fields_small)
            x, y = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))

            grid_output_x = deform_fields[:, 0, :, :]
            grid_output_y = deform_fields[:, 1, :, :]

            x_resample = x[None] + grid_output_x 
            y_resample = y[None] + grid_output_y

            resamp_grid = torch.stack((y_resample, x_resample), dim=-1).float()
            
            resampled = F.grid_sample(moving, resamp_grid) #apply deformation field to input
            loss, l2_loss, variational = criterion(resampled, fixed, deform_fields_small)
            loss.backward()
            optimizer.step()
            summary_train.add_scalar("Loss", loss, epoch * len(trainloader) + i)
            summary_train.add_scalar("L2 Loss", l2_loss, epoch * len(trainloader) + i)
            summary_train.add_scalar("Variational Loss", variational, epoch * len(trainloader) + i)

            # print statistics
            running_loss.append(loss.item())
            running_l2.append(l2_loss.item())
            running_variational.append(variational.item())
            if i % 50 == 0:
                max_to_plot = 5
                step = epoch * len(trainloader) + i
                summary_train.add_images("moving", moving[:max_to_plot], step)
                summary_train.add_images("resampled", resampled[:max_to_plot], step)
                summary_train.add_images("fixed", fixed[:max_to_plot], step)
                deform_fields = (deform_fields + 2) / 4
                summary_train.add_images("deformation field x", deform_fields[:max_to_plot, 0:1], step)
                summary_train.add_images("deformation field y", deform_fields[:max_to_plot, 1:2], step)
    #     summary_train.add_scalar("Loss", torch.mean(running_loss), epoch)
    #     summary_train.add_scalar("L2 Loss", torch.mean(running_l2), epoch)
    #     summary_train.add_scalar("Variational Loss", torch.mean(running_loss), epoch)
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, np.mean(running_loss)))

       
    print('Finished Training') 
    
    #Save Model
    #PATH = '/heartvista_simple_1'
    #torch.save(net.state_dict(), PATH)

if __name__ == "__main__":
    main()
