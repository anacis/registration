#! /usr/bin/env python

import heapq
import os.path
import re
import json
from glob import glob
from operator import itemgetter
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import UFData, normalize
from momentum_model import MomentumModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from network import SimpleNet


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--datadir', "--dd",
                      help='Directory contains 2D images.')
    parser.add_argument('--logdir', type=str,
                      help='The directory containing the model checkpoints')
    parser.add_argument('-i', '--checkpoint', dest='checkpoint',
                      help='Specific checkpoint to load')
    # parser.add_argument('--force_rebuild_memory', action="store_true", default=False,
    #                   help='The path to store (or load) the memory bank.')
    # parser.add_argument('--ip', '--inputpatch', dest='inputpatch', default=False,
    #                   help='inputpatch')
    # parser.add_argument('-f', '--features', dest='features',
    #                   default=128, type='int', help='Features for the training (default 128)')
    # parser.add_argument('-p', '--patch', dest='patch',
    #                   default=20, type='int', help='Patch number for a single image (default 20)')
    parser.add_argument('-n', '--neighbors', dest='neighbors',
                      default=20, type=int, help='neighbors to visualize')
    parser.add_argument('-g', '--gpu_id', type=int,
                      help='use cuda')
    parser.add_argument('--fastmri',  action='store_true', default=False, help='Test on Knee FastMRI dataset')
    # parser.add_argument('--save', dest='save', action="store_true", help='save')
    parser.add_argument('--use_magnitude',  action='store_true', default=False, help='Test on magnitude of data only')
    parser.add_argument('--norm', default=0.95, type=float,
                      help='normalization percentile')
    parser.add_argument('--temperature', '--temp', default=1.00, type=float,
                      help='temperature parameter default: 1')

    parser_arguments = parser.parse_args()
    return parser_arguments

def load_train_parameters(arguments):
    """Restore arguments from last saved json file"""
    params_dir = os.path.join(arguments.logdir, "params")
    params_paths = sorted(glob(os.path.join(params_dir, "params_*")))
    if not len(params_paths):
        raise RuntimeError("Tried to load parameters (to evaluate), but no params file was found!")
    params = Namespace(**{**vars(arguments), **json.load(open(params_paths[-1], "r"))})
    params.logdir = arguments.logdir
    params.gpu_id = arguments.gpu_id
    params.checkpoint = arguments.checkpoint
    return params


def complex2channels(image, dim=0):
        """Convert single from complex to channels"""
        return torch.cat((image.real, image.imag), dim=dim).float()

def normalize_numpy(img):
    """Normalize image in range 0-1"""
    return (img - img.min())/(img.max() - img.min())

class EmbeddingsTopK:

    def __init__(self, reference_embedding, reference_image,
                 embedding_index, centers_x, centers_y, max_to_keep=20):

        self.embedding = reference_embedding
        self.heap = []
        self.max_to_keep = max_to_keep

        self.centers_x = centers_x
        self.centers_y = centers_y

        self.patch = self.get_patch(torch.from_numpy(np.abs(reference_image)), embedding_index)

    def push(self, score, image_patch):
        # Avoid exact same score with some random in very low decimal points
        heapq.heappush(self.heap, (-score + np.random.random() * 1e-10, image_patch))

        if len(self.heap) >= self.max_to_keep:
            worst_index, _ = self.worst_score
            del self.heap[worst_index]
            try:
                heapq.heapify(self.heap)  # Should not be needed if we are deleting a leaf
            except Exception:
                print("uh oh")
                print([item[0] for item in self.heap])

                a = 1

    def push_batch(self, batch_embeddings, batch_images):
        batch_size, feature_dim, rows, cols = batch_embeddings.shape
        #TODO: inspect what self.embedding and batch embeddings and batch scores looks like
        #(also distribution)
        batch_scores = torch.sum(self.embedding[None, :, None, None] * batch_embeddings, dim=1)
        # for i, batch in enumerate(batch_scores):
        #     plt.figure()
        #     plt.imshow(batch.detach().cpu())
        #     plt.savefig(os.path.join(save_dir, f"batch_{i}.png"))

        batch0 = batch_embeddings[:, 0, 0, 0]
        batch1 =  batch_embeddings[0, :, 0, 0]
        batch2 =  batch_embeddings[0, 0, :, 0]
        batch3 =  batch_embeddings[0, 0, 0, :]
        
        dropout = 0.0
        if dropout:
            # Add some random sampling to avoid getting many images from same patient and area
            # Keep something like 50% of the patches (however, we will get different results every time)
            batch_scores = dropout * torch.nn.functional.dropout(batch_scores, p=dropout, inplace=True)
        batch_images = batch_images.norm(dim=1)  # Complex to Abs

        # view = torch.topk(batch_scores.view(-1), self.max_to_keep)

        for value, index in zip(*torch.topk(batch_scores.view(-1), self.max_to_keep)):
            image_index = index // (rows * cols)
            index = index - image_index * rows * cols
            row, col = index // cols, index % cols
            assert batch_scores[image_index, row, col] == value

            self.push(value.item(), self.get_patch(batch_images[image_index], (row, col)).numpy())

    @property
    def worst_score(self):
        index, score = None, -1
        if self.heap:
            # Largest since we store `-score` in our minHeap
            try:
                index, item = heapq.nlargest(1, enumerate(self.heap), key=itemgetter(1))[0]
            except Exception as error:
                print("uh oh")
                a = 1
            score = - item[0]
        return index, score

    def get_patch(self, image, embedding_index, width=47):
        center = np.array([self.centers_x[embedding_index[0]], self.centers_y[embedding_index[1]]])
        start, end = center - width // 2, center + width // 2 + 1
        start_from_0 = np.maximum(start, 0)
        patch = image[start_from_0[0]:end[0], start_from_0[1]:end[1]]

        top_left_pad = - np.minimum(start, 0)
        patch = torch.nn.functional.pad(patch, (top_left_pad[1], 0, top_left_pad[0], 0))

        bottom_right_pad = [width - p for p in patch.shape]
        patch = torch.nn.functional.pad(patch, (0, bottom_right_pad[1], 0, bottom_right_pad[0]))

        return patch.detach().cpu()

    def plot(self):
        sorted_heap = heapq.nsmallest(len(self.heap), self.heap)
        patches = np.concatenate([normalize_numpy(np.abs(item[1])) for item in sorted_heap], axis=1)
        
        white_line = np.ones_like(self.patch)[:, :4] * patches.max()
        patches = np.concatenate([normalize_numpy(np.abs(self.patch)), white_line, patches], axis=1)

        plt.imshow(patches, interpolation="lanczos", cmap="gray", vmin=0, vmax=1)
        plt.title("              " + "     ".join([f"{-item[0]:.4f}" for item in sorted_heap]))
        plt.axis("off")

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = get_args()
    #Check if a param file already exists
    if not args.logdir:
        raise RuntimeError("Need to specify a log directory")
    params_dir = os.path.join(args.logdir, "params")
    params_paths = sorted(glob(os.path.join(params_dir, "params_*")))
    if len(params_paths) > 0:
        print("Existing Arguments")
        args = load_train_parameters(args)

    print(f"Existing Args: {args}")

    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id is not None else 'cpu')

    if not args.checkpoint:
        print(os.path.join(args.logdir, 'checkpoints/*'))
        checkpoint = sorted(glob(os.path.join(args.logdir, 'checkpoints/*')),
                            key=lambda x: int(re.match(".*[a-z]+(\d+).pth", x).group(1)))[-1]
    else:
        checkpoint = args.checkpoint

    checkpoint_number = re.match(".*[a-z]+(\d+).pth", checkpoint).group(1)
    ksnet = MomentumModel(SimpleNet, magnitude=args.use_magnitude, temperature=args.temperature)
    # ksnet = SimpleResNet()

    random_weights = False
    if random_weights:
        print("Using random weights!!!")
        memory_bank_path = "Random Weights"
    else:
        print(f"Loading checkpoint from: {checkpoint}")
        # Loading on cpu before transferring to model
        ksnet.load_state_dict(torch.load(checkpoint, "cpu")["state_dict"])

        # THIS DIDNT WORK - got rubbish predictions
        # net_keyword = "target_network"
        # loaded_dict = torch.load(checkpoint, "cpu")["state_dict"]
        # simple_net_dict = {f"{net_keyword}.{k}": v for k, v in ksnet.state_dict().items()}
        # ksnet.load_state_dict({k[len(net_keyword) + 1:]: v for k, v in simple_net_dict.items() if k in loaded_dict})

    ksnet = ksnet.target_network
    ksnet.to(device)
    ksnet.eval()

    save_dir = os.path.join(args.logdir, "results")
    os.makedirs(save_dir, exist_ok=True)


    # Get corresponding embeddings of patches we want. Convert to admissible centers given the network.
    # jump = reference_image.shape[-2] / (all_reference_embeddings.shape[-2] + 1)
    # assert jump % 1 == 0
    # jump = int(jump)
    # start = jump

    # Run every image on the dataset, find the closest embeddings to each reference patch embeddings and keep the top 20
    dataset = UFData(args.datadir, max_offset=(0, 0), magnitude=args.use_magnitude, device=device, fastmri=args.fastmri, normalization=args.norm)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False, num_workers=20)

    start, jump = 23, 8

    # Run reference image through network
    if args.fastmri:
        input_paths = [
            "/home/asdegoyeneche/UFLoss/fastmri_val_images/file1000153_4.npy",
            "/home/asdegoyeneche/UFLoss/fastmri_val_images/file1001059_2.npy",
        ]
        patch_centers = [(300, 120), (310, 200), (280, 290),
                         (450, 120), (95, 180), (200, 220)]
    else:
        input_paths = [
            "/mikRAID/frank/data/cube_knees/valid_img_slices/18_100.npy",
            "/mikRAID/frank/data/cube_knees/valid_img_slices/18_130.npy",
            # "/mikRAID/frank/data/cube_knees/valid_img_slices/17_100.npy",
            # "/mikRAID/frank/data/cube_knees/valid_img_slices/17_130.npy",
        ]
        patch_centers = [(174, 170), (150, 95), (48, 175),
                         (174, 60), (95, 150), (200, 175)]

    reference_image_size = dataset[0][0].shape
    centers_x = np.arange(start, reference_image_size[-2], jump)
    centers_y = np.arange(start, reference_image_size[-1], jump)

    reference_embeddings = []
    reference_embeddings_2 = [] #second plot comparing references against themselves and against their augmentations
    for i, input_path in enumerate(input_paths):

        reference_image = np.load(input_path)
        reference_image_tensor = torch.from_numpy(reference_image[None])

        if args.use_magnitude:
            reference_image_tensor = torch.tensor(np.abs(reference_image),
                                              dtype=torch.float).to(device)[None, None]
        else:
            reference_image_tensor = complex2channels(reference_image_tensor)[None]
            
        #TODO: we should automatically apply the preprocessing to the reference image
        percent= args.norm
        reference_image_tensor = normalize(reference_image_tensor, percent)
        reference_image_tensor = reference_image_tensor.to(device)
        
        all_reference_embeddings = ksnet(reference_image_tensor)[0].squeeze(0)

        for j, patch_center in enumerate(np.array(patch_centers)):
            embedding_indices = np.round((patch_center - start) / jump).astype(np.int32)
            embedding = all_reference_embeddings[:, embedding_indices[0], embedding_indices[1]]

            # Have a set/heap of patches with scores for each reference patch
            reference_embeddings.append(EmbeddingsTopK(embedding, reference_image, embedding_indices,
                                                       centers_x, centers_y))
            reference_embeddings_2.append(EmbeddingsTopK(embedding, reference_image, embedding_indices,
                                                       centers_x, centers_y))
            #if we are looking at the first image, append it a second time for plot 2
            if i == 0 and j == 0:
                reference_embeddings_2.append(EmbeddingsTopK(embedding, reference_image, embedding_indices,
                                                       centers_x, centers_y))

    with torch.no_grad():
        for images in tqdm(dataloader, "Batch"):
            for image in images[:1]:  # image1 and image2 with different augmentations
                image = image.to(device)
                embeddings = ksnet(image)[0]
                for reference in reference_embeddings:
                    reference.push_batch(embeddings, image)
                    embeddings.detach()
                    image.detach()

        #compare all the reference embeddings to each other
        main_reference = reference_embeddings[0]
        embeddings = []
        images = [] 
        for reference in reference_embeddings_2[1:]:
            embeddings.append(reference.embedding)
            images.append(reference.patch)
        main_reference.push_patch(embeddings, images)

    
    num_references = len(reference_embeddings)
    plt.figure(figsize=(15, 1.3 * num_references))
    for index, reference in enumerate(reference_embeddings):
        plt.subplot(num_references, 1, index + 1)
        reference.plot()

    plt.suptitle(f"{checkpoint}")
    plt.tight_layout()

    filename = f"ckpt{checkpoint_number}_{os.path.splitext(os.path.basename(input_path))[0]}"
    if "fastmri" in args.datadir:
        filename = f"{filename}_fastmri"
    plt.savefig(os.path.join(save_dir, f"{filename}.png"))
    plt.show()




