import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
from imageio import imwrite

def main():
    root = '/home/server31/minyeong_workspace/FL-bench/data/cifar10_iid/raw'
    cid_list = list(range(5))
    try:
        indices = pkl.load(open('/home/server31/minyeong_workspace/FL-bench/data/cifar10_iid/indices.pkl', 'rb'))
    except:
        partition = pkl.load(open('/home/server31/minyeong_workspace/FL-bench/data/cifar10_iid/partition.pkl', 'rb'))
        indices = {int(k) : np.concatenate([partition['data_indices'][int(k)]['train'], partition['data_indices'][int(k)]['test']]) for k in range(len(partition['data_indices']))}
    all_train_idx = []
    all_test_idx = []
    data = np.load(os.path.join(root, 'xdata.npy'))
    
    for cid in cid_list:
        train_dir = os.path.join(root, f'{cid}', 'train')
        test_dir = os.path.join(root, f'{cid}', 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        idx_list = indices[cid]
        # train_idx_list = idx_list[:int(0.9 * len(idx_list))]
        # test_idx_list = idx_list[int(0.9 * len(idx_list)):]
        train_idx_list = idx_list[:-2000]
        test_idx_list = idx_list[-2000:]
        print(f'train_idx_list: {len(train_idx_list)}')
        print(f'test_idx_list: {len(test_idx_list)}')
        # while True:
        #     continue
        for idx in tqdm(train_idx_list):
            save_path = os.path.join(train_dir, '{:05d}.png'.format(idx))
            imwrite(save_path, data[idx])
        for idx in tqdm(test_idx_list):
            save_path = os.path.join(test_dir, '{:05d}.png'.format(idx))
            imwrite(save_path, data[idx])
        all_train_idx.append(np.random.permutation(train_idx_list)[:10000])
        all_test_idx.append(np.random.permutation(test_idx_list)[:10000])
    
    # all_train_dir = os.path.join(root, 'all_50000', 'train')
    # all_test_dir = os.path.join(root, 'all_50000', 'test')
    # os.makedirs(all_train_dir, exist_ok=True)
    # os.makedirs(all_test_dir, exist_ok=True)
    # all_train_idx = np.concatenate(all_train_idx)
    # all_test_idx = np.concatenate(all_test_idx)
    # for j, idx in tqdm(enumerate(all_train_idx)):
    #     save_path = os.path.join(all_train_dir, '{:05d}.png'.format(j))
    #     imwrite(save_path, data[idx])
    # for j, idx in tqdm(enumerate(all_test_idx)):
    #     save_path = os.path.join(all_test_dir, '{:05d}.png'.format(j))
    #     imwrite(save_path, data[idx])
    
    print('done')
    

if __name__=='__main__':
    main()
        
        
            
        
    