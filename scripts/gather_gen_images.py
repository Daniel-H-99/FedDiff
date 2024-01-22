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

    all_local_files = []

    try:
        indices = pkl.load(open('/home/server32/minyeong_workspace/FL-bench/data/femnist/indices.pkl', 'rb'))
    except:
        partition = pkl.load(open('/home/server32/minyeong_workspace/FL-bench/data/femnist/partition.pkl', 'rb'))
        indices = {int(k) : np.concatenate([partition['data_indices'][int(k)]['train'], partition['data_indices'][int(k)]['test']]) for k in range(len(partition['data_indices']))}

    num_data = np.array([len(indices[k]) for k in sorted(list(indices.keys()))])
    total_data = num_data.sum()
    portion = np.array(num_data / total_data)
    num_samples_to_agg = (N * portion).astype('int32')
    num_samples_to_agg[-1] = N - num_samples_to_agg[:-1].sum()

    # N_per_client = [N // N_client] * N_client
    # N_per_client.append(N - sum(N_per_client))

    cid_list = list(range(376))

    for cid in cid_list:
        cid_dir = os.path.join(root, 'local', f'{cid}')
        files = os.listdir(cid_dir)
        all_local_files.extend(np.random.permutation([os.path.join(cid_dir, f) for f in files])[:num_samples_to_agg[cid]])

    for j, f in tqdm(enumerate(all_local_files)):
        shutil.copy(f, os.path.join(local_save_dir, '{:05d}.png'.format(j)))
    
    print(f'done')


if __name__=='__main__':
    todo()