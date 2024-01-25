import pickle
import sys
import json
import os
import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
import time
from typing import Dict, List, OrderedDict
import wandb
import torch.multiprocessing as mp
import torch
import subprocess

torch.multiprocessing.set_sharing_strategy('file_system')
# def work(model):
#     print(f'trainer id: {model.model.base.num_classes}')
#     return model.model.base.num_classes



def work_train(task):
    trainer, params, clients, epochs, verbose, current_epoch = task
    delta_list = []
    weight_list = []
    res_list = []
    for param, client_id, epoch in zip(params, clients, epochs):
        client_local_params = param
        (
            delta,
            weight,
            res,
        ) = trainer.train(
            client_id=client_id,
            local_epoch=epoch,
            new_parameters=client_local_params,
            verbose=verbose,
            epoch=current_epoch
        )
        delta_list.append(delta)
        weight_list.append(weight)
        res_list.append((client_id, res))
    
    return (trainer.state_dict(), delta_list, weight_list, res_list)



def work_test(task):
    correct_before = []
    correct_after = []
    loss_before = []
    loss_after = []
    num_samples = []
    fid_local =[]
    fid_global = []
    fid_adv = []
    
    trainer, params, clients, epoch, _ = task

    for param, client_id in zip(params, clients):
        client_local_params = param
        stats = trainer.test(client_id, client_local_params, epoch)

        correct_before.append(stats["before"]["test_correct"])
        correct_after.append(stats["after"]["test_correct"])
        loss_before.append(stats["before"]["test_loss"])
        loss_after.append(stats["after"]["test_loss"])
        num_samples.append(stats["before"]["test_size"])
        fid_local.append(stats["fid_local"])  
        fid_global.append(stats["fid_global"])  
        fid_adv.append(stats["fid_adv"])  
        
    return (correct_before, correct_after, loss_before, loss_after, num_samples, fid_local, fid_global, fid_adv, clients)


def work_fid(task):
    correct_before = []
    correct_after = []
    loss_before = []
    loss_after = []
    num_samples = []
    fid_local =[]
    fid_global = []
    fid_adv = []
    trainer, params, clients, epoch, _, save_dir = task

    for param, client_id in zip(params, clients):
        client_local_params = param
        stats = trainer.calc_fid(client_id, client_local_params, epoch, save_dir)

        correct_before.append(stats["before"]["test_correct"])
        correct_after.append(stats["after"]["test_correct"])
        loss_before.append(stats["before"]["test_loss"])
        loss_after.append(stats["after"]["test_loss"])
        num_samples.append(stats["before"]["test_size"])
        fid_local.append(stats["fid_local"])  
        fid_global.append(stats["fid_global"])  
        fid_adv.append(stats["fid_adv"])  
        
    return (correct_before, correct_after, loss_before, loss_after, num_samples, fid_local, fid_global, fid_adv, clients)


mp.set_start_method('spawn', True)
# from multiprocessing import Pool, set_start_method
# set_start_method('fork')
from concurrent import futures

 
import torch
from rich.console import Console
from rich.progress import track

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

sys.path.append(PROJECT_DIR.as_posix())
ABC=1
from src.utils.tools import (
    OUT_DIR,
    Logger,
    fix_random_seed,
    parse_config_file,
    trainable_params,
    get_best_device,
)

from src.utils.models import get_model_arch
from src.client.feddiff import FedDiffClient

    
def get_feddiff_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="custom",
        choices=[
            "lenet5",
            "avgcnn",
            "alex",
            "2nn",
            "squeeze0",
            "squeeze1",
            "res18",
            "res34",
            "res50",
            "res101",
            "res152",
            "dense121",
            "dense161",
            "dense169",
            "dense201",
            "mobile2",
            "mobile3s",
            "mobile3l",
            "efficient0",
            "efficient1",
            "efficient2",
            "efficient3",
            "efficient4",
            "efficient5",
            "efficient6",
            "efficient7",
            "custom",
        ],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "mnist_niid2",
            "pathmnist",
            "pathmnist_class0",
            "path_niid",
            "cifar10",
            "cifar10_class0",
            "cifar10_niid2",
            "cifar10_niid3",
            "cifar10_iid",
            "cifar100",
            "organa",
            "organa_niid",
            "synthetic",
            "femnist",
            "emnist",
            "fmnist",
            "celeba",
            "medmnistS",
            "medmnistA",
            "medmnistC",
            "covid19",
            "svhn",
            "usps",
            "tiny_imagenet",
            "cinic10",
            "domain",
        ],
        default="cifar10",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-jr", "--join_ratio", type=float, default=0.1)
    parser.add_argument("-ge", "--global_epoch", type=int, default=30000)
    parser.add_argument("-le", "--local_epoch", type=int, default=1)
    parser.add_argument("-fe", "--finetune_epoch", type=int, default=0)
    parser.add_argument("-tg", "--valid_gap", type=int, default=10000)
    parser.add_argument("-eg", "--test_gap", type=int, default=5)
    parser.add_argument("-ee", "--eval_test", type=int, default=1)
    parser.add_argument("-er", "--eval_train", type=int, default=0)
    parser.add_argument("-lr", "--local_lr", type=float, default=2e-4)
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-vg", "--verbose_gap", type=int, default=1)
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument("-v", "--visible", type=int, default=0)
    parser.add_argument("--global_testset", type=int, default=0)
    parser.add_argument("--straggler_ratio", type=float, default=0)
    parser.add_argument("--straggler_min_local_epoch", type=int, default=1)
    parser.add_argument("--external_model_params_file", type=str, default="")
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--save_log", type=int, default=1)
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--save_fig", type=int, default=1)
    parser.add_argument("--save_metrics", type=int, default=1)
    parser.add_argument("--save_gap", type=int, default=5)
    parser.add_argument("--viz_win_name", type=str, required=False)
    parser.add_argument("-cfg", "--config_file", type=str, default="")
    parser.add_argument("--check_convergence", type=int, default=1)
    parser.add_argument("--personal_tag", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    return parser


class FedDiffServer:
    def __init__(
        self,
        algo: str = "FedDiff",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
        for_eval=False,
        wandb_pj=None
    ):
        self.wandb_pj = wandb_pj
        self.for_eval = for_eval
        self.args = get_feddiff_argparser().parse_args() if args is None else args
        self.algo = algo
        self.unique_model = unique_model
        if len(self.args.config_file) > 0 and os.path.exists(
            Path(self.args.config_file).absolute()
        ):
            self.args = parse_config_file(self.args)
        fix_random_seed(self.args.seed)
        with open(PROJECT_DIR / "data" / self.args.dataset / "args.json", "r") as f:
            self.args.dataset_args = json.load(f)

        # get client party info
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")
        self.train_clients: List[int] = partition["separation"]["train"][:10]
        self.test_clients: List[int] = partition["separation"]["test"][:10]

        # self.client_num: int = partition["separation"]["total"]
        self.client_num: int = 10

        # init model(s) parameters
        self.device = get_best_device(self.args.use_cuda)
        print(f'server device: {self.device}')

        self.device = 'cpu'

        # get_model_arch() would return a class depends on model's name,
        # then init the model object by indicating the dataset and calling the class.
        # Finally transfer the model object to the target device.
        self.model = get_model_arch(model_name=self.args.model)(
            dataset=self.args.dataset, device='cpu'
        )

        
        # self.model.check_avaliability()

        # client_trainable_params is for pFL, which outputs exclusive model per client
        # global_params_dict is for traditional FL, which outputs a single global model
        self.client_trainable_params: List[List[torch.Tensor]] = None
        self.global_params_dict: OrderedDict[str, torch.Tensor] = None

        random_init_params, self.trainable_params_name = trainable_params(
            self.model, detach=True, requires_name=True, personal_tag=args.personal_tag, cpu=True
        )
        

        self.global_params_dict = OrderedDict(
            zip(self.trainable_params_name, random_init_params)
        )

        
        # print(f'trainable params: {self.trainable_params_name}')
        # while True:
        #     continue
        if (
            not self.unique_model
            and self.args.external_model_params_file
            and os.path.isfile(self.args.external_model_params_file)
        ):
            assert False
            # load pretrained params
            self.global_params_dict = torch.load(
                self.args.external_model_params_file, map_location=self.device
            )
        else:
            self.client_trainable_params = [
                trainable_params(self.model, detach=True) for _ in self.train_clients
            ]


        # system heterogeneity (straggler) setting
        self.clients_local_epoch: List[int] = [self.args.local_epoch] * self.client_num
        if (
            self.args.straggler_ratio > 0
            and self.args.local_epoch > self.args.straggler_min_local_epoch
        ):
            straggler_num = int(self.client_num * self.args.straggler_ratio)
            normal_num = self.client_num - straggler_num
            self.clients_local_epoch = [self.args.local_epoch] * (
                normal_num
            ) + random.choices(
                range(self.args.straggler_min_local_epoch, self.args.local_epoch),
                k=straggler_num,
            )
            random.shuffle(self.clients_local_epoch)


        self.NUM_TRAINER = 5
        self.NUM_GPU = 8
        
        # To make sure all algorithms run through the same client sampling stream.
        # Some algorithms' implicit operations at client side may disturb the stream if sampling happens at each FL round's beginning.
        self.client_sample_stream = [
            random.sample(
                self.train_clients, max(1, len(self.train_clients))
                # self.train_clients, max(1, int(self.client_num * self.args.join_ratio))
            )
            for _ in range(self.args.global_epoch)
        ]
        self.selected_clients: List[int] = []
        self.current_epoch = 0
        # For controlling behaviors of some specific methods while testing (not used by all methods)
        self.test_flag = False

        # variables for logging
        if not os.path.isdir(OUT_DIR / self.algo) and (
            self.args.save_log or self.args.save_fig or self.args.save_metrics
        ):
            os.makedirs(OUT_DIR / self.algo, exist_ok=True)
        if not os.path.isdir(OUT_DIR / self.algo / 'checkpoints'
        ):
            os.makedirs(OUT_DIR / self.algo / 'checkpoints', exist_ok=True)

        if self.args.visible:
            from visdom import Visdom

            self.viz = Visdom()
            if self.args.viz_win_name is not None:
                self.viz_win_name = self.args.viz_win_name
            else:
                self.viz_win_name = (
                    f"{self.algo}"
                    + f"_{self.args.dataset}"
                    + f"_{self.args.global_epoch}"
                    + f"_{self.args.local_epoch}"
                )
        self.client_stats = {i: {} for i in self.train_clients}
        self.metrics = {
            "train_before": [],
            "train_after": [],
            "test_before": [],
            "test_after": [],
        }
        stdout = Console(log_path=False, log_time=False)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.save_log,
            logfile_path=OUT_DIR / self.algo / f"{self.args.dataset}_log.html",
        )
        self.test_results: Dict[int, Dict[str, str]] = {}
        self.train_progress_bar = track(
            range(self.current_epoch, self.args.global_epoch), "[bold green]Training...", console=stdout
        )

        self.logger.log("=" * 20, "ALGORITHM:", self.algo, "=" * 20)
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))


        # init trainer
        self.trainers = []
        # def create_client(model, args, i):
        #     model = get_model_arch(model_name=args.model)(
        #         dataset=args.dataset
        #     )
        #     trainer = FedDiffClient(
        #         # deepcopy(self.model), self.args, self.logger, f'cuda:{NUM_GPU - NUM_TRAINER + i}', i
        #         model, args, None, 'cpu', i
        #     )
        #     while True:
        #         continue
        # p = Process(target=f, args=('bob',))

        
        
        

        if default_trainer:
            self.trainers = [FedDiffClient(
                # deepcopy(self.model), self.args, OUT_DIR / self.algo / f"{self.args.dataset}_trainderid{i}_log.html", f'cuda:{i % 2}', i
                deepcopy(self.model), self.args, OUT_DIR / self.algo / f"{self.args.dataset}_trainderid{i}_log.html", f'cuda:{self.NUM_GPU - self.NUM_TRAINER + i}', i
            ) if not self.for_eval else FedDiffClient(
                deepcopy(self.model), self.args, OUT_DIR / self.algo / f"{self.args.dataset}_trainderid{i}_log.html", f'cuda:{i % (self.NUM_GPU - self.NUM_TRAINER)}', i
            ) for i in range(self.NUM_TRAINER)]
            # print(f'type: {type(self.trainers[0].state_dict)}')
            # while True:
            #     continue
            # for tr in self.trainers:
            #     tr.model.share_memory()

            # processes = []
            # for rank in NUM_TRAINER:
            #     p = mp.Process(work, args=(self.trainers[rank]))
            #     p.start()
        else:
            assert False

        if self.args.ckpt is not None:
            server_path = self.args.ckpt + '.pt'
            server_d = torch.load(server_path)
            self.global_params_dict.update(server_d)
            for i in range(self.NUM_TRAINER):
                trainer_ckpt_name = f'{self.args.ckpt}_trainer{i}'
                self.trainers[i].load_trainer(trainer_ckpt_name)
            print(f'loaded from ckpt {self.args.ckpt}')
            # while True:
            #     continue
        del self.model
        torch.cuda.empty_cache()

        if not self.for_eval:
            self.stdout = open(OUT_DIR / self.algo / 'test_stdout.log', 'w')
            self.stderr = open(OUT_DIR / self.algo / 'test_stderr.log', 'w')
        
        self.proc = None
            
    def get_epoch(self, f):
        return int(f.split('_')[2])
    
    def update_last_optimizer_checkpoint(self, save_dir):
        opt_checkpoints = [f for f in os.listdir(save_dir) if 'opt' in f]
        epoch_opt_dict = {}
        for f in opt_checkpoints:
            epoch_opt_dict[self.get_epoch(f)] = epoch_opt_dict.get(self.get_epoch(f), []) + [f]
        assert len(epoch_opt_dict) <= 2, f'{len(epoch_opt_dict)}'
        if len(epoch_opt_dict) == 2:
            assert len(list(epoch_opt_dict.values())[0]) == len(list(epoch_opt_dict.values())[1]), f'{len(list(epoch_opt_dict.values())[0])} vs {len(list(epoch_opt_dict.values())[1])}'
            keys = sorted(list(epoch_opt_dict.keys()))
            for f in epoch_opt_dict[keys[0]]:
                os.remove(os.path.join(save_dir, f))
    
    def save_trainers(self, e, save_dir, before=False):
        for trainer in self.trainers:
            trainer.save_trainer(e, save_dir, before=False)
        self.update_last_optimizer_checkpoint(save_dir)
        
    def train(self):
        """The Generic FL training process"""
        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            if (E + 1) % self.args.save_gap == 0:
                save_flag = True
            else:
                save_flag = False
            
            # if save_flag:
            #     model_name = (
            #         f"{self.args.dataset}_{E + 1}_{self.args.model}.pt"
            #     )
            #     save_dir = OUT_DIR / self.algo / 'checkpoints'
            #     if self.unique_model:
            #         torch.save(
            #             self.client_trainable_params, save_dir / model_name
            #         )
            #     else:
            #         torch.save(self.global_params_dict, save_dir / model_name)
            #     self.trainer.save_client(E + 1, save_dir, before=True)
                
            self.selected_clients = self.client_sample_stream[E]
            begin = time.time()
            self.train_one_round()

            end = time.time()
            # self.log_info()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )
                
            if save_flag:
                save_dir = OUT_DIR / self.algo / 'checkpoints'
                model_name = (
                    f"{self.args.dataset}_{E + 1}_{self.args.model}.pt"
                )
                if self.unique_model:
                    torch.save(
                        self.client_trainable_params, save_dir / model_name
                    )
                else:
                    torch.save(self.global_params_dict, save_dir / model_name)
                self.save_trainers(E + 1, save_dir, before=False)
                
            if (E + 1) % self.args.test_gap == 0:
                save_dir = OUT_DIR / self.algo / 'checkpoints' 
                ckpt = save_dir / f"{self.args.dataset}_{E + 1}_{self.args.model}"
                save_img_dir = OUT_DIR / self.algo / 'images_fid' 
                self.test(ckpt, save_img_dir)

        self.logger.log(
            f"{self.algo}'s average time taken by each global epoch: {int(avg_round_time // 60)} m {(avg_round_time % 60):.2f} s."
        )


    # def task_client(self, client_id)
    
    def client_location(self, client_id):
        return client_id % len(self.trainers)

    def generate_task(self, client_ids):
        tasks = [(self.trainers[i], [], [], [], ((self.current_epoch + 1) % self.args.verbose_gap) == 0, self.current_epoch) for i in range(len(self.trainers))]
        for cid in client_ids:
            tasks[self.client_location(cid)][1].append(self.generate_client_params(cid))
            tasks[self.client_location(cid)][2].append(cid)
            tasks[self.client_location(cid)][3].append(self.clients_local_epoch[cid])
        return tasks
    
    def generate_test_task(self, client_ids):
        tasks = [(self.trainers[i], [], [], self.current_epoch, ((self.current_epoch + 1) % self.args.verbose_gap) == 0) for i in range(len(self.trainers))]
        for cid in client_ids:
            tasks[self.client_location(cid)][1].append(self.generate_client_params(cid))
            tasks[self.client_location(cid)][2].append(cid)

        return tasks

    def generate_fid_task(self, client_ids, epoch, save_dir):
        tasks = [(self.trainers[i], [], [], epoch, ((self.current_epoch + 1) % self.args.verbose_gap) == 0, save_dir) for i in range(len(self.trainers))]
        for cid in client_ids:
            tasks[self.client_location(cid)][1].append(self.generate_client_params(cid))
            tasks[self.client_location(cid)][2].append(cid)

        return tasks

    
    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        delta_cache = []
        weight_cache = []
        res_cache = []
        # with futures.ThreadPoolExecutor() as executor:
        #     results = executor.map(foo, [(self.trainer, 1)])
        # print(f'res: {[v for v in results]}')
        # while True:
        #     continue

        # def foo(a):
        #     print(f'ABC: {ABC}')
        #     # t = torch.randn(3).to(f'cuda:{ag[1]}')
        #     # print(f't.device : {t.device}')
        #     # print(f'ag args: {ag.args}')
        #     return a
        # with Pool(2) as pool:
        #     res = pool.map(foo, [0, 1])
        # print(f'done')
        # while True:
        #     continue
            
        # print(f'res: {res}')
        # while True:
        #     continue

        start_time = time.time()

        tasks = self.generate_task(self.selected_clients)
        # print(f'tasks: {tasks}')
        # while True:
        #     continue

        # with futures.ThreadPoolExecutor(8) as executor:
        #     results = executor.map(work, tasks)

        with mp.Pool(self.NUM_TRAINER) as pool:
            res = pool.map(work_train, tasks)

        res = list(res)
        
        print(f'aggregating')
        
        for s, d, w, r in res:
            self.trainers[s["trainer_id"]].load_state_dict(s)
            delta_cache.extend(d)
            weight_cache.extend(w)
            res_cache.extend(r)
        wandb.log({"epoch": self.current_epoch})
        befores = []
        afters = []
        loss_logs = []
        for cid, r in res_cache:
            self.client_stats[cid][self.current_epoch] = r
            befores.append(r["before"]["test_loss"])
            afters.append(r["after"]["test_loss"])
            loss_logs.append(r["loss_log"])
            wandb.log({f'test_loss_before_client_{cid}': r["before"]["test_loss"]})
            wandb.log({f'test_loss_after_client_{cid}': r["after"]["test_loss"]})
            wandb.log({f'train_loss_client_{cid}': r["loss_log"]})
            if ((self.current_epoch + 1) % self.args.verbose_gap) == 0:
                log = r["log_string"]
                self.logger.log(log)
        befores = sum(befores) / len(befores)
        afters = sum(afters) / len(afters)
        loss_logs = sum(loss_logs) / len(loss_logs)
        wandb.log({f'test_loss_before_avg': befores})
        wandb.log({f'test_loss_after_avg': afters})
        wandb.log({f'loss_logs_avg': loss_logs})
        # wandb.log({"test loss before avg": befores, "test loss after avg": afters, "train loss avg": loss_logs})
        
        end_time = time.time()
        # print(f'used time: {end_time - start_time} seconds')
        # print(f'updates: {self.trainers[0].personal_params_dict.keys()}')
        # while True:
        #     continue
        # print(f'delta samples: {delta_cache[:2]}')
        # print(f'weight samples: {weight_cache[:2]}')
        # print(f'delta cache size: {len(delta_cache)}')
        # print(f'delta cache size: {len(weight_cache)}')
        # print(f'used time: {end_time - start_time} seconds')


        
        # for client_id in self.selected_clients:
        #     client_local_params = self.generate_client_params(client_id)
        #     (
        #         delta,
        #         weight,
        #         self.client_stats[client_id][self.current_epoch],
        #     ) = self.trainers[0].train(
        #         client_id=client_id,
        #         local_epoch=self.clients_local_epoch[client_id],
        #         new_parameters=client_local_params,
        #         verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
        #     )
        #     delta_cache.append(delta)
        #     weight_cache.append(weight)
            # print(f'updates: {self.trainers[0].personal_params_dict.keys()}')
            # while True:
            #     continue
        # print(f'delta device: {list(delta.values())[0].device}')
        # print(f'weight device: {weight.device}')
        # while True:
        #     continue
            
        self.aggregate(delta_cache, weight_cache)

    def calc_fid(self, epoch):
        """The function for testing FL method's output (a single global model or personalized client models)."""
        self.test_flag = True
        loss_before, loss_after = [], []
        correct_before, correct_after = [], []
        num_samples = []
        fid_local, fid_global, fid_adv = [], [], []
        cids = []
        save_dir = OUT_DIR / self.algo / 'images_fid'
        os.makedirs(save_dir, exist_ok=True)
        tasks = self.generate_fid_task(self.test_clients, epoch, save_dir)
        # print(f'tasks: {[len(v[1]) for v in tasks]}')
        # while True:
        #     continue
        # print(f'tasks: {tasks}')
        # while True:
        #     continue

        with mp.Pool(self.NUM_TRAINER) as pool:
            res = pool.map(work_fid, tasks)
        res = list(res)
        
        # print(f'res shape: {len(res)}')
        # print(f'clients: {[v[-1] for v in res]}')
        # while True:
        #     continue
        for cb, ca, lb, la, ns, fl, fg, fa, cid in res:
            correct_before.extend(cb)
            correct_after.extend(ca)
            loss_before.extend(lb)
            loss_after.extend(la)
            num_samples.extend(ns)
            fid_local.extend(fl)
            fid_global.extend(fg)
            fid_adv.extend(fa)
            cids.extend(cid)
        print(f'res shape: {len(res)}')
        print(f'fid_adv: {[v[-2] for v in res]}')

        # for client_id in self.test_clients:
        #     client_local_params = self.generate_client_params(client_id)
        #     stats = self.trainer.test(client_id, client_local_params)

        #     correct_before.append(stats["before"]["test_correct"])
        #     correct_after.append(stats["after"]["test_correct"])
        #     loss_before.append(stats["before"]["test_loss"])
        #     loss_after.append(stats["after"]["test_loss"])
        #     num_samples.append(stats["before"]["test_size"])

        loss_before = torch.tensor(loss_before)
        loss_after = torch.tensor(loss_after)
        correct_before = torch.tensor(correct_before)
        correct_after = torch.tensor(correct_after)
        num_samples = torch.tensor(num_samples)



        loss_before = (num_samples * loss_before).sum() / num_samples.sum()
        loss_after = (num_samples * loss_after).sum() / num_samples.sum()
        
        
        log_dict = {"epoch": self.current_epoch + 1, "epoch_loss": loss_before}
        
        fid_local_dict = {f'fid_local_client_{k}': v for k, v in zip(cids, fid_local)}
        fid_global_dict = {f'fid_global_client_{k}': v for k, v in zip(cids, fid_global)}
        fid_adv_dict = {f'fid_adv_client_{k}': v for k, v in zip(cids, fid_adv)}
        fid_local_avg = sum(fid_local) / len(fid_local)
        fid_global_avg = sum(fid_global) / len(fid_global)
        fid_adv_avg = sum(fid_adv) / len(fid_adv)
        fid_avg_dict = {
            "fid_local_avg": fid_local_avg,
            "fid_global_avg": fid_global_avg,
            "fid_adv_avg": fid_adv_avg,
        }
        log_dict.update(fid_local_dict)
        log_dict.update(fid_global_dict)
        log_dict.update(fid_adv_dict)
        log_dict.update(fid_avg_dict)

        
        # while True:
        #     continue
        self.test_flag = False
        return log_dict
        
        
    def test(self, checkpoint, save_dir):
        if self.proc is not None:
            self.proc.wait()
            self.proc = None
        pj = self.wandb_pj.project
        id = self.wandb_pj.id
        # while True:
        #     print(f'checkpoint: {checkpoint}')
        test_cmd = ['python', 'test_fid_api.py', 'feddiff', f'{checkpoint}', f'{save_dir}', f'{pj}', f'{id}', '-d', 'cifar10_niid3', '--join_ratio', '1.0']
        self.proc = subprocess.Popen(args=test_cmd, stdout=self.stdout, stderr=self.stderr)
        # print(f'waiting')
        # self.proc.wait()
    # def test(self):
    #     """The function for testing FL method's output (a single global model or personalized client models)."""
    #     self.test_flag = True
    #     loss_before, loss_after = [], []
    #     correct_before, correct_after = [], []
    #     num_samples = []
    #     fid_local, fid_global, fid_adv = [], [], []
    #     cids = []
        
    #     tasks = self.generate_test_task(self.test_clients)
    #     # print(f'tasks: {tasks}')
    #     # while True:
    #     #     continue

    #     with mp.Pool(self.NUM_TRAINER) as pool:
    #         res = pool.map(work_test, tasks)
    #     res = list(res)
        
    #     # print(f'res shape: {len(res)}')
    #     # print(f'clients: {[v[-1] for v in res]}')
    #     # while True:
    #     #     continue
    #     for cb, ca, lb, la, ns, fl, fg, fa, cid in res:
    #         correct_before.extend(cb)
    #         correct_after.extend(ca)
    #         loss_before.extend(lb)
    #         loss_after.extend(la)
    #         num_samples.extend(ns)
    #         fid_local.extend(fl)
    #         fid_global.extend(fg)
    #         fid_adv.extend(fa)
    #         cids.extend(cid)
    #     print(f'res shape: {len(res)}')
    #     print(f'fid_adv: {[v[-2] for v in res]}')

    #     # for client_id in self.test_clients:
    #     #     client_local_params = self.generate_client_params(client_id)
    #     #     stats = self.trainer.test(client_id, client_local_params)

    #     #     correct_before.append(stats["before"]["test_correct"])
    #     #     correct_after.append(stats["after"]["test_correct"])
    #     #     loss_before.append(stats["before"]["test_loss"])
    #     #     loss_after.append(stats["after"]["test_loss"])
    #     #     num_samples.append(stats["before"]["test_size"])

    #     loss_before = torch.tensor(loss_before)
    #     loss_after = torch.tensor(loss_after)
    #     correct_before = torch.tensor(correct_before)
    #     correct_after = torch.tensor(correct_after)
    #     num_samples = torch.tensor(num_samples)



    #     loss_before = (num_samples * loss_before).sum() / num_samples.sum()
    #     loss_after = (num_samples * loss_after).sum() / num_samples.sum()
        
    #     self.test_results[self.current_epoch + 1] = {
    #         "loss": "{:.4f} -> {:.4f}".format(
    #             loss_before,
    #             loss_after,
    #         ),
    #         "accuracy": "{:.2f}% -> {:.2f}%".format(
    #             correct_before.sum() / num_samples.sum() * 100,
    #             correct_after.sum() / num_samples.sum() * 100,
    #         ),
    #     }
        
        
    #     log_dict = {"epoch": self.current_epoch + 1, "epoch_loss": loss_before}
        
    #     fid_local_dict = {f'fid_local_client_{k}': v for k, v in zip(cids, fid_local)}
    #     fid_global_dict = {f'fid_global_client_{k}': v for k, v in zip(cids, fid_global)}
    #     fid_adv_dict = {f'fid_adv_client_{k}': v for k, v in zip(cids, fid_adv)}
    #     fid_local_avg = sum(fid_local) / len(fid_local)
    #     fid_global_avg = sum(fid_global) / len(fid_global)
    #     fid_adv_avg = sum(fid_adv) / len(fid_adv)
    #     fid_avg_dict = {
    #         "fid_local_avg": fid_local_avg,
    #         "fid_global_avg": fid_global_avg,
    #         "fid_adv_avg": fid_adv_avg,
    #     }
    #     log_dict.update(fid_local_dict)
    #     log_dict.update(fid_global_dict)
    #     log_dict.update(fid_adv_dict)
    #     log_dict.update(fid_avg_dict)

    #     self.test_results[self.current_epoch + 1].update(log_dict)
        
    #     wandb.log(log_dict)
    #     # while True:
    #     #     continue
    #     self.test_flag = False

    @torch.no_grad()
    def update_client_params(self, client_params_cache: List[List[torch.Tensor]]):
        """
        The function for updating clients model while unique_model is `True`.
        This function is only useful for some pFL methods.

        Args:
            client_params_cache (List[List[torch.Tensor]]): models parameters of selected clients.

        Raises:
            RuntimeError: If unique_model = `False`, this function will not work properly.
        """
        if self.unique_model:
            for i, client_id in enumerate(self.selected_clients):
                self.client_trainable_params[client_id] = client_params_cache[i]
        else:
            raise RuntimeError(
                "FL system don't preserve params for each client (unique_model = False)."
            )

    def generate_client_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        """
        This function is for outputting model parameters that asked by `client_id`.

        Args:
            client_id (int): The ID of query client.

        Returns:
            OrderedDict[str, torch.Tensor]: The trainable model parameters.
        """
        if self.unique_model:
            return OrderedDict(
                zip(self.trainable_params_name, self.client_trainable_params[client_id])
            )
        else:
            return self.global_params_dict

    @torch.no_grad()
    def aggregate(
        self,
        delta_cache: List[OrderedDict[str, torch.Tensor]],
        weight_cache: List[int],
        return_diff=True,
    ):
        """
        This function is for aggregating recevied model parameters from selected clients.
        The method of aggregation is weighted averaging by default.

        Args:
            delta_cache (List[List[torch.Tensor]]): `delta` means the difference between client model parameters that before and after local training.

            weight_cache (List[int]): Weight for each `delta` (client dataset size by default).

            return_diff (bool): Differnt value brings different operations. Default to True.
        """
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        # print(f'weights: {weights}')
        if return_diff:
            delta_list = [list(delta.values()) for delta in delta_cache]
            aggregated_delta = [
                torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
                for diff in zip(*delta_list)
            ]
            # print(f'aggrega deltas: {aggregated_delta[:10]}')
            for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
                param.data -= diff
        else:
            for old_param, zipped_new_param in zip(
                self.global_params_dict.values(), zip(*delta_cache)
            ):
                old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                    dim=-1
                )
        
        # self.model.load_state_dict(self.global_params_dict, strict=False)

    def check_convergence(self):
        """This function is for checking model convergence through the entire FL training process."""
        for label, metric in self.metrics.items():
            if len(metric) > 0:
                self.logger.log(f"Convergence ({label}):")
                acc_range = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
                min_acc_idx = 10
                max_acc = 0
                for E, acc in enumerate(metric):
                    for i, target in enumerate(acc_range):
                        if acc >= target and acc > max_acc:
                            self.logger.log(
                                "{} achieved {}%({:.2f}%) at epoch: {}".format(
                                    self.algo, target, acc, E
                                )
                            )
                            max_acc = acc
                            min_acc_idx = i
                            break
                    acc_range = acc_range[:min_acc_idx]

    def log_info(self):
        """This function is for logging each selected client's training info."""
        split = self.args.dataset_args["split"]
        label = {"sample": "test", "user": "train"}[split]
        correct_before = torch.tensor(
            [
                self.client_stats[i][self.current_epoch]["before"][f"{label}_correct"]
                for i in self.selected_clients
            ]
        )
        correct_after = torch.tensor(
            [
                self.client_stats[i][self.current_epoch]["after"][f"{label}_correct"]
                for i in self.selected_clients
            ]
        )
        loss_before = torch.tensor(
            [
                self.client_stats[i][self.current_epoch]["before"][f"{label}_loss"]
                for i in self.selected_clients
            ]
        )
        loss_after = torch.tensor(
            [
                self.client_stats[i][self.current_epoch]["after"][f"{label}_loss"]
                for i in self.selected_clients
            ]
        )
        
        num_samples = torch.tensor(
            [
                self.client_stats[i][self.current_epoch]["before"][f"{label}_size"]
                for i in self.selected_clients
            ]
        )

        acc_before = (
            loss_before.sum(dim=-1, keepdim=True) / num_samples.sum() * 100.0
        ).item()
        acc_after = (
            loss_after.sum(dim=-1, keepdim=True) / num_samples.sum() * 100.0
        ).item()
        self.metrics[f"{label}_before"].append(acc_before)
        self.metrics[f"{label}_after"].append(acc_after)

        if self.args.visible:
            self.viz.line(
                [acc_before],
                [self.current_epoch],
                win=self.viz_win_name,
                update="append",
                name=f"{label}(before)",
                opts=dict(
                    title=self.viz_win_name,
                    xlabel="Communication Rounds",
                    ylabel="Accuracy",
                ),
            )
            self.viz.line(
                [acc_after],
                [self.current_epoch],
                win=self.viz_win_name,
                update="append",
                name=f"{label}(after)",
            )

    def run(self):
        """The comprehensive FL process.

        Raises:
            RuntimeError: If `trainer` is not set.
        """
        begin = time.time()
        if self.trainers is None:
            raise RuntimeError(
                "Specify your unique trainer or set `default_trainer` as True."
            )

        if self.args.visible:
            self.viz.close(win=self.viz_win_name)

        self.train()
        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algo}'s total running time: {int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.logger.log(
            "=" * 20, self.algo, "TEST RESULTS:", "=" * 20, self.test_results
        )

        if self.args.check_convergence:
            self.check_convergence()

        self.logger.close()

        if self.args.save_fig:
            import matplotlib
            from matplotlib import pyplot as plt

            matplotlib.use("Agg")
            linestyle = {
                "test_before": "solid",
                "test_after": "solid",
                "train_before": "dotted",
                "train_after": "dotted",
            }
            for label, acc in self.metrics.items():
                if len(acc) > 0:
                    plt.plot(acc, label=label, ls=linestyle[label])
            plt.title(f"{self.algo}_{self.args.dataset}")
            plt.ylim(0, 100)
            plt.xlabel("Communication Rounds")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(
                OUT_DIR / self.algo / f"{self.args.dataset}.jpeg", bbox_inches="tight"
            )
        if self.args.save_metrics:
            import pandas as pd
            import numpy as np

            accuracies = []
            labels = []
            for label, acc in self.metrics.items():
                if len(acc) > 0:
                    accuracies.append(np.array(acc).T)
                    labels.append(label)
            pd.DataFrame(np.stack(accuracies, axis=1), columns=labels).to_csv(
                OUT_DIR / self.algo / f"{self.args.dataset}_acc_metrics.csv",
                index=False,
            )
        # save trained model(s)
        if self.args.save_model:
            model_name = (
                f"{self.args.dataset}_{self.args.global_epoch}_{self.args.model}.pt"
            )
            if self.unique_model:
                torch.save(
                    self.client_trainable_params, OUT_DIR / self.algo / model_name
                )
            else:
                torch.save(self.global_params_dict, OUT_DIR / self.algo / model_name)


if __name__ == "__main__":
    server = FedDiffServer()
    server.run()
