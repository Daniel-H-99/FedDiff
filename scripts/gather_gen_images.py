import os
import shutil
import numpy as np
from tqdm import tqdm


def todo(root=f'/home/server36/minyeong_workspace/FL-bench/images_fid/10', N=5000, N_client=5):
    # save_dir = '/home/server36/minyeong_workspace/FL-bench/images_fid/10'
    local_save_dir = os.path.join(root, 'local', 'all')
    if os.path.exists(local_save_dir):
        shutil.rmtree(local_save_dir)
    os.makedirs(local_save_dir)

    all_local_files = []
    files = os.listdir(os.path.join(root, 'local'))
    cids = []
    for f in files:
        try:
            cids.append(int(f))
        except:
            pass
    N_client = len(cids)
    N_per_client = [N // N_client] * N_client
    N_per_client.append(N - sum(N_per_client))
    for cid in cids:
        cid_dir = os.path.join(root, 'local', f'{cid}')
        files = os.listdir(cid_dir)
        all_local_files.extend(np.random.permutation([os.path.join(cid_dir, f) for f in files])[:N_per_client[cids.index(cid)]])

    for j, f in tqdm(enumerate(all_local_files)):
        shutil.copy(f, os.path.join(local_save_dir, '{:05d}.png'.format(j)))
    
    print(f'done')


if __name__=='__main__':
    todo()