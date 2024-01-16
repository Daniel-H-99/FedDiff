import json
import numpy as np
import os
import importlib
import sys
import inspect
from pathlib import Path
import wandb 
from pytorch_fid import fid_score, inception
from src.utils.ddpm.ddpm_torch.metrics.fid_score import calculate_privacy_given_paths, calculate_privacy2_given_paths
import pickle as pkl
from scripts.gather_gen_images import todo
import argparse
sys.path.append(Path(__file__).parent.joinpath("src/server").absolute().as_posix())

# image_fid_dir = '/home/server36/minyeong_workspace/FL-bench/images_fid'
image_fid_dir = '/home/server36/minyeong_workspace/FL-bench/out_cifar10_niid3_vqfed_trial1/FedDiff/images_fid'
true_image_dir = '/home/server36/minyeong_workspace/FL-bench/data/cifar10_niid3/raw'

CID=0
def init_wandb():
    wandb.init(project='fids', name=f'condphoenix_cifar10_niid3_client{CID}')
    
def load_models(cls, args, ckpt_name):
    args.ckpt = ckpt_name
    server = cls(args=args)
    return server

def calc_privacy(path_1, path_2):
    
    privacy = calculate_privacy_given_paths([path_1, path_2],
                                            batch_size=50,
                                            device=f'cuda:{CID}',
                                            dims=3 * 28 * 28,
                                            num_workers=8)
    return privacy
    
def calc_privacy2(path_1, path_2):
    
    privacy = calculate_privacy2_given_paths([path_1, path_2],
                                            batch_size=50,
                                            device=f'cuda:{CID}',
                                            dims=2048,
                                            num_workers=8)
    return privacy


def calc_fid(path_1, path_2):
    fid = fid_score.calculate_fid_given_paths([path_1, path_2],
                                            batch_size=50,
                                            device=f'cuda:{CID}',
                                            dims=2048,
                                            num_workers=8)
    return fid

def calc_fid_dict_external(src_path, tgt_path):
    init_wandb()
    output = {}
    print(f'trying src path: {src_path} \n tgt path: {tgt_path}')
    output[f'fid'] = calc_fid(src_path, tgt_path)
    # res[f'local_global_client_{client_id}'] = calc_fid(syn_local_path, true_global_path)
    # res[f'global_global_client_{client_id}'] = calc_fid(syn_local_path, true_global_path)
    # res[f'global_global_client_{client_id}'] = calc_fid(syn_global_path, true_global_path)
    wandb.log(output, step=0)

    return output

def calc_fid_dict(checkpoints):
    init_wandb()
    output = {}
    for ckpt in checkpoints:
        res = {}
        epoch = int(os.path.basename(ckpt).split('_')[2])
        for client_id in range(CID, CID + 1):
            print(f'trying client id: {client_id}')
            syn_local_path = os.path.join(image_fid_dir, f'{epoch}', 'local', f'{client_id}')
            syn_tmp_path = os.path.join(image_fid_dir, f'{epoch}', 'tmp')
            syn_all_path = os.path.join(image_fid_dir, f'{epoch}', 'local', 'all')
            true_local_path = os.path.join(true_image_dir, f'{client_id}', 'train')
            syn_global_path = os.path.join(image_fid_dir, f'{epoch}', 'global', f'{client_id}')
            true_global_path = os.path.join(true_image_dir, 'all_50000', 'train')
            # res[f'local_local_client_{client_id}'] = calc_fid(syn_local_path, true_local_path)
            # res[f'local_global_client_{client_id}'] = calc_fid(syn_local_path, true_global_path)
            res[f'global_global_client_{client_id}'] = calc_fid(syn_all_path, true_global_path)
            # res[f'global_global_client_{client_id}'] = calc_fid(syn_global_path, true_global_path)
        wandb.log(res, step=epoch)
        output[epoch] = res
    return output

def calc_privacy_dict(checkpoints):
    init_wandb()
    output = {}
    for ckpt in checkpoints:
        res = {}
        epoch = int(os.path.basename(ckpt).split('_')[2])
        for client_id in range(CID, CID + 1):
            print(f'trying client id: {client_id}')
            syn_local_path = os.path.join(image_fid_dir, f'{epoch}', 'local', f'{client_id}')
            train_local_path = os.path.join(true_image_dir, f'{client_id}', 'train')
            test_local_path = os.path.join(true_image_dir, f'{client_id}', 'test')
            syn_global_path = os.path.join(image_fid_dir, f'{epoch}', 'global', f'{client_id}')
            train_global_path = os.path.join(true_image_dir, 'all', 'train')
            test_global_path = os.path.join(true_image_dir, 'all', 'test')
            res[f'local_train_client_{client_id}'] = calc_privacy(syn_local_path, train_local_path)
            res[f'local_test_client_{client_id}'] = calc_privacy(syn_local_path, test_local_path)
            res[f'global_train_client_{client_id}'] = calc_privacy(syn_global_path, train_global_path)
            res[f'global_test_client_{client_id}'] = calc_privacy(syn_global_path, test_global_path)
        wandb.log(res, step=epoch)
        output[epoch] = res
    return output

def calc_privacy2_dict(checkpoints):
    init_wandb()
    output = {}
    for ckpt in checkpoints:
        res = {}
        epoch = int(os.path.basename(ckpt).split('_')[2])
        for client_id in range(CID, CID + 1):
            print(f'trying client id: {client_id}')
            syn_local_path = os.path.join(image_fid_dir, f'{epoch}', 'local', f'{client_id}')
            syn_tmp_path = os.path.join(image_fid_dir, f'{epoch}', 'tmp')
            # train_local_path = os.path.join(true_image_dir, f'{client_id}', 'train')
            # test_local_path = os.path.join(true_image_dir, f'{client_id}', 'test')
            syn_global_path = os.path.join(image_fid_dir, f'{epoch}', 'global', f'{client_id}')
            train_global_path = os.path.join(true_image_dir, 'all', 'train')
            test_global_path = os.path.join(true_image_dir, 'all', 'test')
            # res[f'local_train_client_{client_id}'] = calc_privacy2(syn_local_path, train_local_path)
            # res[f'local_test_client_{client_id}'] = calc_privacy2(syn_local_path, test_local_path)
            res[f'global_train_client_{client_id}'] = calc_privacy2(syn_global_path, train_global_path)
            res[f'global_test_client_{client_id}'] = calc_privacy2(syn_global_path, test_global_path)
        wandb.log(res, step=epoch)
        output[epoch] = res
    return output


            
def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Need to assign a method. Run like `python main.py <method> [args ...]`, e.g., python main.py fedavg -d cifar10 -m lenet5`"
        )
    method = sys.argv[1]
    args_list = sys.argv[2:]

    module = importlib.import_module(method)
    try:
        get_argparser = getattr(module, f"get_{method}_argparser")
    except:
        fedavg_module = importlib.import_module("fedavg")
        get_argparser = getattr(fedavg_module, "get_fedavg_argparser")
    parser = get_argparser()
    module_attributes = inspect.getmembers(module, inspect.isclass)
    server_class = [
        attribute
        for attribute in module_attributes
        if attribute[0].lower() == method + "server"
    ][0][1]

    args = parser.parse_args(args_list)
    
    # server = load_models(server_class, args, None)
    
    # print(f'loaded server')
    
    ckpt_dir = f'/home/server36/minyeong_workspace/FL-bench/out_cifar10_niid3_vqfed_trial1/FedDiff/checkpoints'
    files = sorted(list(set([int(f.split('_')[2]) for f in os.listdir(ckpt_dir) ])))
    ckpt_name_list = [os.path.join(ckpt_dir, f"cifar10_niid3_{f}_custom") for f in files if (85 < f) and (f <= 100)]
    
    # print(f'ckpt_name_list: {ckpt_name_list}')
    # while True:
    #     continue
    
    for ckpt_name in ckpt_name_list:
        server = load_models(server_class, args, ckpt_name)
        log = server.calc_fid(int(os.path.basename(ckpt_name).split('_')[2]))
        todo(os.path.join(image_fid_dir, f'{int(os.path.basename(ckpt_name).split("_")[2])}'), N=50000)
        print(f'{log}')
    
    
    # cifar_src_path = '/home/server36/minyeong_workspace/ddpm-torch/images/eval/cifar10/cifar10_2040_ddim'
    # # cifar_src_path = '/home/server36/minyeong_workspace/ddpm-torch/tmp'
    # # cifar_tgt_path = '/home/server36/minyeong_workspace/FL-bench/data/cifar10/raw/all/train'
    # cifar_tgt_path = '/home/server36/minyeong_workspace/FL-bench/data/cifar10/raw/_all/train'
    # fid_dict = calc_fid_dict_external(cifar_src_path, cifar_tgt_path)

    # with open(f'tested_fid_fed_cifar10_client.pkl', 'wb') as f:
    #     pkl.dump(fid_dict, f)


    
    fid_dict = calc_fid_dict(ckpt_name_list)
    with open(f'tested_fid_vqfed_cifar10_niid3_client_{CID}.pkl', 'wb') as f:
        pkl.dump(fid_dict, f)

        
    # privacy_dict = calc_privacy_dict(ckpt_name_list)
    # with open(f'tested_privacy_fed_class0_client_{CID}.pkl', 'wb') as f:
    #     pkl.dump(privacy_dict, f)

    # privacy_dict = calc_privacy2_dict(ckpt_name_list)
    # with open(f'tested_privacy2_fed_cifar10_class0_client_{CID}.pkl', 'wb') as f:
    #     pkl.dump(privacy_dict, f)
        
    print(f'done')

    

# def task_checkpoint(ckpt_name, client_id=0):
#     label_dist = {}
#     with open('/home/server36/minyeong_workspace/FL-bench/data/pathmnist/all_stats.json', 'r') as f:
#         d = json.load(f)
#         for k, v in d.items():
#             if k == 'sample per client':
#                 continue
#             dist = np.zeros(9)
#             for _k, _v in v['y'].items():
#                 dist[int(_k)] = _v
#             dist = dist / dist.sum()
#             label_dist[int(k)] = dist
#     # print(f'dist 0: {label_dist[0]}')
#     label_dist = label_dist
#     base, eval, img_dir = export_trainer(device='cuda', eval_device='cuda', eval_total_size=3000)
#     # fid_adv = eval.eval(base.sample_fn, self.label_dist[self.client_id], [f'/home/server36/minyeong_workspace/FL-bench/data/pathmnist/fid_stats_pathmnist_client{self.client_id}.npz'], is_leader=True, adv=True)['fid'][0]
#     fid_local = eval.eval(base.sample_fn, label_dist[client_id], [f'/home/server36/minyeong_workspace/FL-bench/data/pathmnist/fid_stats_pathmnist_client{client_id}.npz'], is_leader=True)['fid'][0]
#     # fid_global = self.model.evaluator.eval(self.model.base.sample_fn, np.ones_like(self.label_dist[self.client_id]) / len(self.label_dist[self.client_id]), [f'/home/server36/minyeong_workspace/FL-bench/data/pathmnist/fid_stats_pathmnist.npz'], is_leader=True)['fid'][0]

#     # eval_results = [fid_local, fid_global, fid_adv]
#     print(f'evaluated local fid: {fid_local}')
    
if __name__=="__main__":
    main()