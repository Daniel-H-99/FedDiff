import os
import shutil
import numpy as np
from tqdm import tqdm
import pickle as pkl

def todo(root=f'/home/server36/minyeong_workspace/FL-bench/images_fid/10', N=5000, N_client=5):
    # save_dir = '/home/server36/minyeong_workspace/FL-bench/images_fid/10'
    local_save_dir = os.path.join(root, 'local', 'all')
    if os.path.exists(local_save_dir):
        shutil.rmtree(local_save_dir)
    os.makedirs(local_save_dir)


    # cid_list = list(range(376))
    try:
        indices = pkl.load(open('/home/server36/minyeong_workspace/FL-bench/data/femnist/indices.pkl', 'rb'))
    except:
        partition = pkl.load(open('/home/server36/minyeong_workspace/FL-bench/data/femnist/partition.pkl', 'rb'))
        indices = {int(k) : np.concatenate([partition['data_indices'][int(k)]['train'], partition['data_indices'][int(k)]['test']]) for k in range(len(partition['data_indices']))}


    files = os.listdir(os.path.join(root, 'local'))
    client_list = []
    for f in files:
        try:
            client_list.append(int(f))
        except:
            pass

    num_data = np.array([len(indices[k]) for k in client_list])
    total_data = num_data.sum()
    portion = np.array(num_data / total_data)
    num_samples_to_agg = (N * portion).astype('int32')
    num_samples_to_agg[-1] = N - num_samples_to_agg[:-1].sum()


    all_local_files = []

    for j, cid in enumerate(client_list):
        cid_dir = os.path.join(root, 'local', f'{cid}')
        files = os.listdir(cid_dir)
        all_local_files.extend(np.random.permutation([os.path.join(cid_dir, f) for f in files])[:num_samples_to_agg[j]])

    for j, f in tqdm(enumerate(all_local_files)):
        shutil.copy(f, os.path.join(local_save_dir, '{:05d}.png'.format(j)))
    
    print(f'done')


if __name__=='__main__':
    todo()