import numpy as np
import os
import cv2
import sys
import shutil
from tqdm import tqdm
from skimage import img_as_ubyte, img_as_float32

def main(name):
    root = 'data'
    npz_name = os.path.join(root, f'{name}.npz')
    save_dir = os.path.join(root, f'{name}')
    raw_dir = os.path.join(save_dir, 'raw')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    os.makedirs(save_dir)
    os.makedirs(raw_dir)

    with np.load(npz_name) as f:
        data = dict(f)

    # print(f'data keys: {data.keys()}')
    # while True:
    #     continue
    
    splits = ['train', 'val', 'test']
    all_data = []
    all_label = []
    for split in splits:
        all_data.append(data[f'{split}_images'])
        all_label.append(data[f'{split}_labels'])
    all_data = np.concatenate(all_data)
    all_label = np.concatenate(all_label)
    if len(all_label.shape) > 1:
        all_label = all_label[:, 0]
    
    if len(all_data.shape) == 3:
        all_data = all_data[:, :, :, None]
    if all_data.shape[3] == 1:
        all_data = all_data.repeat(3, axis=3)
    if all_data.shape[1] != 32:
        _all_data = []
        for img in tqdm(all_data):
            _all_data.append(img_as_ubyte(cv2.resize(img_as_float32(img), (32, 32))))
        all_data = np.stack(_all_data)
    assert len(all_data.shape) == 4

    print(f'xdata type: {all_data.dtype}')
    print(f'ydata type: {all_label.dtype}')
    np.save(os.path.join(raw_dir, 'xdata.npy'), all_data)
    np.save(os.path.join(raw_dir, 'ydata.npy'), all_label)



if __name__=='__main__':
    name = sys.argv[1]
    main(name)