import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/server33/minyeong_workspace/FL-bench')
from data.utils.datasets import PathMNIST

import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import math 
from tqdm import tqdm
import pickle as pkl
from imageio import imread


# *-- VGG model 읽어오기 --*
use_pretrained=True # 학습 된 파라미터 사용
net = models.vgg16(pretrained=use_pretrained)
net.avgpool = torch.nn.AvgPool2d((7, 7))
net.classifier = torch.nn.Identity()
net.eval()
net.cuda()
# 모델 네트워크 구성 출력
print(net)


### loading data###
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, .224, 0.225)
transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.Normalize(mean, std)]
)


# xdata = np.load(os.path.join(data_path, 'xdata.npy'))
# ydata = np.load(os.path.join(data_path, 'ydata.npy'))

# data_path = '/home/server33/minyeong_workspace/FL-bench/images_fid/8/local'
data_path = '/home/server33/minyeong_workspace/FL-bench/data/cifar10/raw'
files = []
data = []
labels = []
cid_list = list(range(20))
for cid in cid_list:
    files_dir = os.path.join(data_path, f'{cid}', 'train')
    cid_files = os.listdir(files_dir)
    files.extend([os.path.join(files_dir, f) for f in cid_files])
    labels.extend([cid] * len(cid_files))

for f in tqdm(files):
    data.append(imread(f))

data = np.stack(data)
# labels = np.array(labels)
# class0_indice = (ydata == 0).nonzero()[0]
# print(f'class0_indices: {class0_indice[:10]}')
# print(f'len of class0: {len(class0_indice)}')
print(f'x data shape: {data.shape}')
print(f'x data min max : {data.min()} {data.max()}')

data = torch.tensor(data).permute(0, 3, 1, 2) / 255.0
batch_size = 32
all_embeds = torch.zeros(0)
for i in tqdm(range(math.ceil(len(data) / batch_size))):
    input = data[batch_size * i : batch_size * (i + 1)]
    input = transform(input).cuda()
    with torch.no_grad():
        embeds = net(input)
    all_embeds = torch.cat([all_embeds, embeds.detach().cpu()])
print(f'embed shape: {all_embeds.shape}')

X = all_embeds

N_MEANS = 7
# kmeans = KMeans(n_clusters=N_MEANS, random_state=0, n_init="auto").fit(X)
# labels = kmeans.predict(X)
# split_indices = {}
# for i in range(N_MEANS):
#     split_indices[i] = class0_indice[(labels == i).nonzero()[0]]

# with open('/home/server36/minyeong_workspace/FL-bench/data/pathmnist_class0/indices.pkl', 'wb') as f:
#     pkl.dump(split_indices, f)


# print(f'num_indices: {[len(split_indices[k]) for k in split_indices.keys()]}')

tsne = TSNE(n_components=2, random_state=0) # 사실 easy 함 sklearn 사용하니..
cluster = np.array(tsne.fit_transform(np.array(X)))
actual = np.array(labels)

plt.figure(figsize=(10, 10))

for i in range(N_MEANS):
    idx = np.where(actual == i)
    plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=i)

plt.legend()
plt.savefig('cifar10_train_fed.png')

# dataset = PathMNIST(
#     root= "data/pathmnist",
#     general_data_transform=general_data_transform,
#     general_target_transform=general_target_transform,
#     train_data_transform=train_data_transform,
#     train_target_transform=train_target_transform,
# )

