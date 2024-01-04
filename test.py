from data.utils.datasets import RawPathMNIST
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.fid import evaluate_fid_score
from sklearn.manifold import TSNE
from src.utils.models import CustomModel
import wandb 
import torch
import argparse
import imageio
import os 
import numpy as np
from matplotlib import pyplot as plt

def plot_tsne(features, save_dir):
    tsne = TSNE(n_components=2, random_state=0)
    clusters = np.array(tsne.fit_transform(features))
    plt.figure(figsize=(10, 10))
    plt.scatter(clusters[:, 0], clusters[:, 1], marker='.')
    plt.savefig(os.path.join(save_dir, 'tsne.png'))
    

def sample_image(model, ref):
    # ref: B x C x H x W
    B = ref.shape[0]
    with torch.no_grad():
        sample = model.sample(B)["gen"]
    return sample

def postprocess(sample, mean, sig):
    # samples: B x 3 x H x W
    # mean, sig: 3
    B = len(sample)
    L = sample.shape[2]
    # print(f'sample shape: {sample.shape}')
    # print(f'sig shape: {sig.shape}')
    # sample = sample.transpose(1, 0).flatten(1) * sig[:, None] + mean[:, None]
    sample = sample.permute(0, 2, 3, 1)
    print(f'min: {sample.min()} vs max: {sample.max()}')
    sample = sample.clamp(min=0, max=1)
    return sample
   
def save_imgs(samples, N=20, save_dir=""):
    samples = samples[:N]
    samples = img_as_ubyte(samples.cpu().numpy())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, img in enumerate(samples):
        imageio.imsave(os.path.join(save_dir, f'{i}.png'), img)
    
def main(args):
    ### Init ###
    print(f'Starting Testing')
    device = 'cuda' if args.cuda else 'cpu'
    
    ### Prepare Experiment ###
    dataset = RawPathMNIST(split="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    model = CustomModel("pathmnist").to(device)
    # model.load_state_dict(torch.load('/home/server40/minyeong_workspace/FL-bench/out_single/FedDiff/checkpoints/pathmnist_100_custom.pt'), strict=False)
    model.load_state_dict(torch.load('/home/server40/minyeong_workspace/FL-bench/out_single/FedDiff/checkpoints/pathmnist_100_custom_client_after.pt')[0], strict=False)
    
    ### Run ###
    batch_size=1
    samples = []
    _samples = []
    z_tildes = []
    dl_iter = iter(dataloader)
    model.eval()
    model.diff.eval()
    for i in tqdm(range(args.num_samples // batch_size)):
        _sample = next(dl_iter)
        _samples.append(_sample["img"].float())
        precision_scope = torch.autocast
        with precision_scope("cuda"):
            with torch.no_grad():
                sample = model.sample_ddim(batch_size, )["gen"]
                z_tilde = model.encode_z(_sample["img"].float().to(device))["z_tilde"].cpu()
        samples.append(sample.detach().cpu().float())
        z_tildes.append(z_tilde.float())
    samples = torch.cat(samples)
    _samples = torch.cat(_samples)
    z_tildes = torch.cat(z_tildes)
    
    samples = postprocess(samples, torch.tensor(dataloader.dataset.mean).float(), torch.tensor(dataloader.dataset.sig).float())
    _samples = postprocess(_samples, torch.tensor(dataloader.dataset.mean).float(), torch.tensor(dataloader.dataset.sig).float())
    
    save_imgs(samples, save_dir=os.path.join("samples", args.name, "gen"))
    save_imgs(_samples, save_dir=os.path.join("samples", args.name, "true"))
    # plot_tsne(z_tildes.numpy(), save_dir=os.path.join("samples", args.name))
    
    print(f'caculating fid...')
    # fid = evaluate_fid_score(samples.numpy(), _samples.numpy())
    # print(f'fid: {fid}')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="vae")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()

    main(args)
    
    

    
        
        
     
    