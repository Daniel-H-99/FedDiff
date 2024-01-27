import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/server36/minyeong_workspace/FL-bench')
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
from torchvision.models import resnet18

# *-- VGG model 읽어오기 --*
model = resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()
model.cuda()


### loading data###
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, .224, 0.225)
transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.Normalize(mean, std)]
)

data_path = 'data/path_niid/raw'
xdata = np.load(os.path.join(data_path, 'xdata.npy'))
ydata = np.load(os.path.join(data_path, 'ydata.npy'))
parition = pkl.load(open('data/path_niid/partition.pkl', 'rb'))
indices = [np.concatenate([p['train'], p['test']]) for p in parition['data_indices']]
train_indices = [idx[:int(0.9 * len(idx))] for idx in indices]
print(f'x data shape: {xdata.shape}')
print(f'x data min max : {xdata.min()} {xdata.max()}')

data = torch.tensor(xdata).permute(0, 3, 1, 2) / 255.0
batch_size = 32
all_embeds = torch.zeros(0)
for i in tqdm(range(math.ceil(len(data) / batch_size))):
    input = data[batch_size * i : batch_size * (i + 1)]
    input = transform(input).cuda()
    with torch.no_grad():
        embeds = model(input)
    all_embeds = torch.cat([all_embeds, embeds.detach().cpu()])
print(f'embed shape: {all_embeds.shape}')

for cid, idx in tqdm(enumerate(train_indices)):
    X = all_embeds[idx]
    N_MEANS = 256
    # N_MEANS = min(32, len(X))
    # centroids = np.zeros(32, X.shape[1])
    # np.save(f'{data_path}/vq_centroid_num_client{cid}.npy', N_MEANS)

    kmeans = KMeans(n_clusters=N_MEANS, random_state=0, n_init="auto").fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    np.save(f'{data_path}/_vq_centroid_label_client{cid}.npy', labels)
    # np.save(f'{data_path}/vq_centroid_client{cid}.npy', centroids)




# labels = kmeans.predict(X)

# split_indices = {}
# for i in range(N_MEANS):
#     split_indices[i] = class0_indice[(labels == i).nonzero()[0]]

# with open('/home/server33/minyeong_workspace/FL-bench/data/cifar10_class0/indices.pkl', 'wb') as f:
#     pkl.dump(split_indices, f)


# print(f'num_indices: {[len(split_indices[k]) for k in split_indices.keys()]}')

# tsne = TSNE(n_components=2, random_state=0) # 사실 easy 함 sklearn 사용하니..
# cluster = np.array(tsne.fit_transform(np.array(X)))
# actual = np.array(labels)

# plt.figure(figsize=(10, 10))

# for i in range(N_MEANS):
#     idx = np.where(actual == i)
#     plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=i)

# plt.legend()
# plt.savefig('class0_dataset_tsne.png')

# dataset = PathMNIST(
#     root= "data/pathmnist",
#     general_data_transform=general_data_transform,
#     general_target_transform=general_target_transform,
#     train_data_transform=train_data_transform,
#     train_target_transform=train_target_transform,
# )

