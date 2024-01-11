import os 
import shutil
import numpy as np


def main():
    root = '/home/server33/minyeong_workspace/FL-bench/data/cifar10/raw'
    N = 50000
    client_list = list(range(20))
    all_train_list = []
    all_test_list = []
    for cid in client_list:
        cid_dir = os.path.join(root, f'{cid}')
        train_dir = os.path.join(cid_dir, 'train')
        test_dir = os.path.join(cid_dir, 'test')
        all_train_list.extend(np.random.permutation([os.path.join(train_dir, f) for f in os.listdir(train_dir)])[:N])
        all_test_list.extend(np.random.permutation([os.path.join(test_dir, f) for f in os.listdir(test_dir)])[:N])
    
    # assert len(all_train_list) == 3000
    # assert len(all_test_list) == 3000

    save_dir = '/home/server33/minyeong_workspace/FL-bench/data/cifar10/raw'
    train_save_dir = os.path.join(save_dir, '_all', 'train')
    test_save_dir = os.path.join(save_dir, '_all', 'test')
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(test_save_dir, exist_ok=True)
    for j, f in enumerate(all_train_list):
        shutil.copy(f, os.path.join(train_save_dir, '{:05d}.png'.format(j)))
    for j, f in enumerate(all_test_list):
        shutil.copy(f, os.path.join(test_save_dir, '{:05d}.png'.format(j)))
        
    print(f'done')
    
    
    
if __name__=='__main__':
    main()
    
