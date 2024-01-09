import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
from imageio import imwrite
def main():
    root = '/home/server36/minyeong_workspace/FL-bench/data/cifar10_class0/raw'
    cid_list = list(range(7))
    indices = pkl.load(open('/home/server36/minyeong_workspace/FL-bench/data/cifar10_class0/indices.pkl', 'rb'))
    all_train_idx = []
    all_test_idx = []
    data = np.load(os.path.join(root, 'xdata.npy'))
    
    for cid in cid_list:
        train_dir = os.path.join(root, f'{cid}', 'train')
        test_dir = os.path.join(root, f'{cid}', 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        idx_list = indices[cid]
        train_idx_list = idx_list[:int(0.9 * len(idx_list))]
        test_idx_list = idx_list[int(0.9 * len(idx_list)):]
        # for idx in tqdm(train_idx_list):
        #     save_path = os.path.join(train_dir, '{:05d}.png'.format(idx))
        #     imwrite(save_path, data[idx])
        # for idx in tqdm(test_idx_list):
        #     save_path = os.path.join(test_dir, '{:05d}.png'.format(idx))
        #     imwrite(save_path, data[idx])
        all_train_idx.append(np.random.permutation(train_idx_list)[:3000])
        all_test_idx.append(np.random.permutation(test_idx_list)[:3000])
    
    all_train_dir = os.path.join(root, 'all', 'train')
    all_test_dir = os.path.join(root, 'all', 'test')
    os.makedirs(all_train_dir, exist_ok=True)
    os.makedirs(all_test_dir, exist_ok=True)
    all_train_idx = np.concatenate(all_train_idx)[:3000]
    all_test_idx = np.concatenate(all_test_idx)[:3000]
    for j, idx in tqdm(enumerate(all_train_idx)):
        save_path = os.path.join(all_train_dir, '{:05d}.png'.format(j))
        imwrite(save_path, data[idx])
    for j, idx in tqdm(enumerate(all_test_idx)):
        save_path = os.path.join(all_test_dir, '{:05d}.png'.format(j))
        imwrite(save_path, data[idx])
    
    print('done')
    

if __name__=='__main__':
    main()
        
        
            
        
    