import os
import re
from glob import glob
from argparse import ArgumentParser, Namespace
from datetime import datetime
import json
import subprocess

import numpy as np
import torch
from dataset import UFData
from momentum_model import MomentumModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from network import SimpleNet


# TODO: Learning rate decay
# TODO: Tune temperature (~0.07?)
# TODO: Maybe sample from memory bank too?
# Done: Sample from encodings?


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--datadir', "--dd",
                      help='Directory contains 2D images.')
    parser.add_argument("-g", '--gpu_id', type=int,
                      help='GPU number, default is None (-g 0 means use gpu 0)')
    parser.add_argument('--logdir', "--ld",
                      help='Directory for saving logs and checkpoints')
    parser.add_argument('-f', '--features', default=128, type=int,
                      help='Dimension of the feature space.')
    parser.add_argument('--learning-rate', '--lr', default=1e-4, type=float,
                      help='learning rate for the model')
    parser.add_argument('--temperature', '--temp', default=1.00, type=float,
                      help='temperature parameter default: 1')
    parser.add_argument('--momentum', default=0.999, type=float,
                      help='Momentum for target network.')
    parser.add_argument('--batchsize', '--bs', dest='batchsize',
                      default=32, type=int, help='batch size for training')
    parser.add_argument('-e', '--epochs', default=200, type=int,
                      help='Number of epochs to train')
    # parser.add_argument('-m', '--model', dest='model',
    #                   default=False, help='load checkpoints')
    parser.add_argument('--use_magnitude', action="store_true", default=False,
                      help='If specified, use image magnitude.')
    # parser.add_argument('-x', '--sx', dest='sx',
    #                   default=256, type='int', help='image dim: x')
    # parser.add_argument('-y', '--sy', dest='sy',
    #                   default=320, type='int', help='image dim: y')
    parser.add_argument('--force_train_from_scratch', '--overwrite', action="store_true",
                      help="If specified, training will start from scratch."
                           " Otherwise, latest checkpoint (if any) will be used")
    parser.add_argument('--fastmri', action="store_true", default=False,
                      help='If specified, use fastmri settings.')
    parser.add_argument('--norm', default=0.95, type=float,
                      help='normalization percentile')

    parser_arguments = parser.parse_args()
    return parser_arguments


class Trainer:

    def __init__(self):

        self.args = get_args()

        #Check if a param file already exists
        params_dir = os.path.join(self.args.logdir, "params")
        if os.path.isdir(params_dir):
            print("Params dir exists")
            params_paths = sorted(glob(os.path.join(params_dir, "params_*")))
            if len(params_paths) > 0:
                print("Existing Arguments")
                self.args = self.load_train_parameters(self.args)

        print(f"log dir {self.args.logdir}")

        self.device = torch.device(f"cuda:{self.args.gpu_id}")
        print("Using device:", self.device)
        print("Using magnitude:", self.args.use_magnitude)

        self.checkpoint_directory = os.path.join(f"{self.args.logdir}", "checkpoints")
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        self.dataset = UFData(self.args.datadir, magnitude=bool(self.args.use_magnitude), device=self.device,
                              fastmri=self.args.fastmri, random_augmentation=True, normalization=self.args.norm)
        self.dataloader = DataLoader(self.dataset, self.args.batchsize, shuffle=True, drop_last=True,
                                     num_workers=self.args.batchsize, pin_memory=True)

        self.model = MomentumModel(SimpleNet, feature_dim=self.args.features, momentum=self.args.momentum,
                                   temperature=self.args.temperature, device=self.device,
                                   magnitude=self.args.use_magnitude)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        self.summary_writer = SummaryWriter(os.path.join(self.args.logdir, "train"), flush_secs=15)

        self.start_epoch = 1
        if not self.args.force_train_from_scratch:
            self.restore_model()
        else:
            input("Training from scratch. Are you sure? (Ctrl+C to kill):")

        # Save command line arguments and other parameters for this run
        self.save_train_parameters(self.args)

    def restore_model(self):
        """Restore latest model checkpoint (if any) and continue training from there."""

        checkpoint_path = sorted(glob(os.path.join(self.checkpoint_directory, "*")),
                                 key=lambda x: int(re.match(".*[a-z]+(\d+).pth", x).group(1)))
        if checkpoint_path:
            checkpoint_path = checkpoint_path[-1]
            print(f"Found saved model at: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
            self.start_epoch = checkpoint["epoch"] + 1  # Start at next epoch of saved model

            print(f"Finish restoring model. Resuming at epoch {self.start_epoch}")

        else:
            print("No saved model found. Training from scratch.")

    def save_model(self, epoch):
        """Save model checkpoint.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """

        torch.save({
            "epoch": epoch,  # Epoch we just finished
            "state_dict": self.model.state_dict(),
            "optimizer_dict": self.optimizer.state_dict()
        }, os.path.join(self.checkpoint_directory, 'ckpt{}.pth'.format(epoch)))

    def save_train_parameters(self, arguments):
        """Save training and network parameters for run.
        Parameters
        ----------
        arguments : Namespace
            The command line arguments for the current run.
        """
        params_dict = vars(arguments)
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        params_dict["date_time"] = timestamp
        params_dict["commit_hash"] = subprocess.check_output(['git', 'log', '-1',
                                                              '--pretty=format:"%H"']).decode().strip('"')
        params_dict["branch"] = subprocess.check_output(['git', 'branch']).decode().split("*")[1].split()[0]
        params_dict["start_epoch"] = self.start_epoch

        params_dir = os.path.join(arguments.logdir, "params")
        if not os.path.isdir(params_dir):
            print("No params dir exists, making a params dir")
            os.mkdir(params_dir)
        with open(os.path.join(params_dir, f"params_{timestamp}.json"), "w") as _file:
            json.dump(params_dict, _file, indent=4)
    
    @staticmethod
    def load_train_parameters(arguments):
        """Restore arguments from last saved json file"""
        params_dir = os.path.join(arguments.logdir, "params")
        params_paths = sorted(glob(os.path.join(params_dir, "params_*")))
        if not len(params_paths):
            raise RuntimeError("Tried to load parameters (to evaluate), but no params file was found!")
        params = Namespace(**{**vars(arguments), **json.load(open(params_paths[-1], "r"))})
        params.logdir = arguments.logdir
        params.gpu_id = arguments.gpu_id
        params.force_train_from_scratch = arguments.force_train_from_scratch
        return params
    
    def train(self):
        """Train the model!"""

        for epoch in tqdm(range(self.start_epoch, self.args.epochs + 1), "Epoch"):

            losses = []
            first_step = True
            self.model.train()
            for index, (images, target_images) in enumerate(tqdm(self.dataloader, "Step")):

                images = images.to(self.device)
                target_images = target_images.to(self.device)

                embeddings, target_embeddings, target_im1_embeddings = self.model(images, target_images)
                logits, labels = self.model.get_logits_labels(embeddings, target_embeddings)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                embeddings.detach()
                target_embeddings.detach()
                target_im1_embeddings.detach()

                with torch.no_grad():
                    self.model.update_target_network()
                    self.model.update_memory_bank(target_embeddings)

                losses.append(loss.item())

                if first_step:
                    first_step = False
                    self.log_images(epoch, images, target_images)
                    self.log_embedding_images(epoch, embeddings, target_im1_embeddings)
                    self.log_embeddings(epoch, images, target_im1_embeddings, "embeddings")
                    self.log_embeddings(epoch, target_images, target_embeddings, "target_embeddings")

            loss = np.mean(losses)
            print(f"\n\n\tEpoch {epoch}. Loss {loss}\n")

            self.summary_writer.add_scalar("Loss", loss, epoch)

            if epoch % 10 == 0:
                self.save_model(epoch)

    def log_embedding_images(self, epoch, images, target_images, max_outputs=5, num_embs=5):
        images = images[:max_outputs]
        target_images = target_images[:max_outputs]

        images /= torch.maximum(images.amax(dim=(1, 2, 3), keepdim=True), torch.ones_like(images))
        target_images /= torch.maximum(target_images.amax(dim=(1, 2, 3), keepdim=True), torch.ones_like(target_images))

        images = images.transpose(1, 3).transpose(2, 3)
        target_images = target_images.transpose(1, 3).transpose(2, 3)
        
        
        for i in range(num_embs):            
            image_group = images[:, i, :, :][:, None].detach().cpu()
            target_image_group = target_images[:, i, :, :][:, None].detach().cpu()
            
            shape = list(image_group.shape)
            shape[2] = 2
            ones = torch.ones(shape)
            both = torch.cat([image_group, ones, target_image_group], dim=2)
            padded = F.pad(both, (0,2,0,0), value=1)

            self.summary_writer.add_images("{} Embedding Im1 Top and Im2 Bot".format(i+1), padded, epoch)    
    
    def log_images(self, epoch, images, target_images, max_outputs=5):
        if not self.args.use_magnitude:
            images = torch.norm(images[:max_outputs], dim=1, keepdim=True)
            target_images = torch.norm(target_images[:max_outputs], dim=1, keepdim=True)

        images = images[:max_outputs]
        target_images = target_images[:max_outputs]
        images /= torch.maximum(images.amax(dim=(1, 2, 3), keepdim=True), torch.ones_like(images))
        target_images /= torch.maximum(target_images.amax(dim=(1, 2, 3), keepdim=True), torch.ones_like(target_images))
                
        self.summary_writer.add_images("Image 1", images, epoch)
        self.summary_writer.add_images("Image 2", target_images, epoch)

    def log_embeddings(self, epoch, images, embeddings, tag="embeddings", patch_size=47, jump=8, sample_rate=0.1):
        if not self.args.use_magnitude:
            images = torch.norm(images, dim=1, keepdim=True)

        # Get embedding indices to collect (and image patches top left corners by multiplying (x, y) by jump)
        index_n, index_x, index_y = torch.meshgrid(torch.arange(embeddings.shape[0]),
                                                   torch.arange(embeddings.shape[1]),
                                                   torch.arange(embeddings.shape[2]))
        index_n, index_x, index_y = torch.flatten(index_n), torch.flatten(index_x), torch.flatten(index_y)
        random_choice = torch.randperm(len(index_n))[:int(sample_rate * len(index_n))]
        index_n, index_x, index_y = index_n[random_choice], index_x[random_choice], index_y[random_choice]

        # Collect images and respective embeddings
        images = torch.stack([images[s_n, :, s_x: s_x + patch_size, s_y: s_y + patch_size]
                              for s_n, s_x, s_y in zip(index_n, index_x * jump, index_y * jump)], 0)
        embeddings = torch.stack([embeddings[s_n, s_x, s_y] for s_n, s_x, s_y in zip(index_n, index_x, index_y)], 0)

        self.summary_writer.add_embedding(embeddings, label_img=images, global_step=epoch, tag=tag)
        self.summary_writer.add_histogram(tag, embeddings)

        embeddings = embeddings[:10, :5][None, None]

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
