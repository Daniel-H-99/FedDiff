from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.fid import evaluate_fid_score
from sklearn.manifold import TSNE
from src.utils.models import CustomModel
from data.utils.datasets import PathMNIST
import wandb 
import torch
import argparse
import imageio
import os 
import numpy as np
import medmnist
import pickle as pkl
import json
from torch.utils.data import DataLoader, Subset

from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from typing import Dict, List, Tuple, Union


def plot_tsne(features, save_dir):
    tsne = TSNE(n_components=2, random_state=0)
    clusters = np.array(tsne.fit_transform(features))
    plt.figure(figsize=(10, 10))
    plt.scatter(clusters[:, 0], clusters[:, 1], marker='.')
    plt.savefig(os.path.join(save_dir, 'tsne.png'))
    

def sample_image(model, ref, c):
    # ref: B x C x H x W
    B = ref.shape[0]
    with torch.no_grad():
        sample = model.sample(B, c.long().to(model.device))["gen"]
    return sample

def postprocess(sample, mean, sig):
    # samples: B x 3 x H x W
    # mean, sig: 3
    B = len(sample)
    L = sample.shape[2]
    # print(f'sample shape: {sample.shape}')
    # print(f'sig shape: {sig.shape}')
    if type(mean) == torch.Tensor:
        sample = sample.transpose(1, 0).flatten(1) * sig[:, None] + mean[:, None]
    else:
        sample = sample.transpose(1, 0).flatten(1) * sig + mean
    sample = sample.view(3, B, L, L).permute(1, 2, 3, 0)
    print(f'min: {sample.min()} vs max: {sample.max()}')
    sample = sample.clamp(min=0, max=1)
    return sample
   
def save_imgs(samples, c, N=20, save_dir=""):
    c = c[:N].long().cpu()
    samples = samples[:N]
    samples = img_as_ubyte(samples.cpu().numpy())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, img in enumerate(samples):
        imageio.imsave(os.path.join(save_dir, f'{i}_c{c[i].item()}.png'), img)
    
def main(args):
    ### Init ###
    print(f'Starting Testing')
    device = 'cuda' if args.cuda else 'cpu'
    batch_size=100
    
    ### Prepare Experiment ###
    # dataset = PathMNIST(split="train")
    # dataloader = DataLoader(dataset, batch_size=100, shuffle=True, drop_last=True)

    ### PathMNIST
    cifar_transform = transforms.Compose(
        [transforms.ToTensor()]
        )
    dataset = medmnist.dataset.PathMNIST(split='train', transform=cifar_transform, download=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # dataset = datasets.CIFAR10(root='./mnist_data/', train=True, transform=cifar_transform, download=True)
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    # ds = 'pathmnist'
    # # load dataset and clients' data indices
    # try:
    #     partition_path =  "/home/server40/minyeong_workspace/FL-bench/data/pathmnist/partition.pkl"
    #     with open(partition_path, "rb") as f:
    #         partition = pkl.load(f)
    # except:
    #     raise FileNotFoundError(f"Please partition {ds} first.")
    # with open( "/home/server40/minyeong_workspace/FL-bench/data/pathmnist/args.json", "r") as f:
    #     dataset_args = json.load(f)

    # data_indices: List[List[int]] = partition["data_indices"]

    # # --------- you can define your own data transformation strategy here ------------
    # general_data_transform = transforms.Compose(
    #     [transforms.Normalize((np.array([0, 0, 0]) * 255.0).tolist(), (np.array([1.0, 1.0, 1.0]) * 255).tolist())]
    # )
    # general_target_transform = transforms.Compose([])
    # train_data_transform = transforms.Compose([])
    # train_target_transform = transforms.Compose([])
    # # --------------------------------------------------------------------------------

    # dataset = PathMNIST(
    #     root="/home/server40/minyeong_workspace/FL-bench/data/pathmnist",
    #     args=dataset_args,
    #     general_data_transform=general_data_transform,
    #     general_target_transform=general_target_transform,
    #     train_data_transform=train_data_transform,
    #     train_target_transform=train_target_transform,
    # )

    # dataloader = None
    # localset = Subset(dataset, indices=[])
    # localset.indices = np.concatenate([data_indices[0]["train"][:], data_indices[0]["test"]])
    # dataloader = DataLoader(localset, 100, num_workers=8)
    
    # print(f'??')
    # while True:
    #     continue

    model = CustomModel('pathmnist').to(device)
    model.load_state_dict(torch.load(os.path.join("out_cvae_all",  "FedDiff", "checkpoints", "pathmnist_300_custom.pt")), strict=False)
    model.load_state_dict(torch.load(os.path.join("out_cvae_all",  "FedDiff", "checkpoints", "pathmnist_300_custom_client_after.pt"))[0], strict=False)
    ### Run ##2
    samples = []
    _samples = []
    labels = []
    z_tildes = []
    dl_iter = iter(dataloader)
    model.eval()
    for i in tqdm(range(args.num_samples // batch_size)):
        _sample = next(dl_iter)
        _samples.append(_sample[0].float())
        with torch.no_grad():
            c = _sample[1].to(device)
            print(f'pre c shape: {c.shape}')
            sample = model.sample(batch_size, c, glob=True)["gen"]
            z_tilde = model(_sample[0].float().to(device), c)["z_tilde"].cpu()
        samples.append(sample.detach().cpu())
        labels.append(c.detach().long().cpu())
        z_tildes.append(z_tilde)
    samples = torch.cat(samples)
    _samples = torch.cat(_samples)
    z_tildes = torch.cat(z_tildes)
    labels = torch.cat(labels)
    
    samples = postprocess(samples, 0, 1)
    _samples = postprocess(_samples, 0, 1)
    
    print(f'name: {args.name}')
    save_imgs(samples, labels, save_dir=os.path.join("samples", args.name, "gen"))
    save_imgs(_samples, labels, save_dir=os.path.join("samples", args.name, "true"))
    plot_tsne(z_tildes.numpy(), save_dir=os.path.join("samples", args.name))
    
    print(f'caculating fid...')
    fid = evaluate_fid_score(samples.numpy(), _samples.numpy())
    print(f'fid: {fid}')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="cvae_all_glob")
    parser.add_argument("--num_samples", type=int, default=3600)
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()

    main(args)
    
    

    
        
        
     
    