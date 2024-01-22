import pickle
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import torch
import math
import json
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

from src.utils.tools import trainable_params, evalutate_model, Logger
from src.utils.models import DecoupledModel
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS
import numpy as np
import os

class FedDiffClient:
    def __init__(
        self,
        model: DecoupledModel,
        args: Namespace,
        logger: Logger,
        device: torch.device,
        trainer_id: int,
        trainer_ckpt_name: str = None
    ):
        self.args = args
        self.device = device
        self.client_id: int = None
        self.trainer_id = trainer_id
        # load dataset and clients' data indices
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        self.data_indices: List[List[int]] = partition["data_indices"]

        # --------- you can define your own data transformation strategy here ------------
        general_data_transform = transforms.Compose(
            [transforms.Normalize((np.array([0.5, 0.5, 0.5]) * 255.0).tolist(), (np.array([0.5, 0.5, 0.5]) * 255).tolist())]
        )
        general_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose([])
        train_target_transform = transforms.Compose([])
        # --------------------------------------------------------------------------------

        self.dataset = DATASETS[self.args.dataset](
            root=PROJECT_DIR / "data" / args.dataset,
            args=args.dataset_args,
            general_data_transform=general_data_transform,
            general_target_transform=general_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )

        self.trainloader: DataLoader = None
        self.testloader: DataLoader = None
        self.trainset: Subset = Subset(self.dataset, indices=[])
        self.testset: Subset = Subset(self.dataset, indices=[])
        self.global_testset: Subset = None
        if self.args.global_testset:
            all_testdata_indices = []
            for indices in self.data_indices:
                all_testdata_indices.extend(indices["test"])
            self.global_testset = Subset(self.dataset, all_testdata_indices)

        # self.predefine_dataset()s
        
        
        self.model = model
        self.model.to(device)

        # self.model.base.device = device

        self.local_epoch = self.args.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        # self.logger = logger
        self.personal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.all_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.personal_tag = args.personal_tag

        self.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach().cpu()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad or ((self.personal_tag is not None) and (self.personal_tag in key))
        }
        self.personal_params_name: List[str] =  list(self.init_personal_params_dict.keys())
        # print(f'pers param dict: {self.init_personal_params_dict.keys()}')
        # while True:
        #     continue
        self.opt_state_dict = {}
        self.data_idx = {}
        # print(f'trainable params: {trainable_params(self.model, requires_name=True)}')
        # while True:
        #     continue
        # print(f'params: {len(trainable_params(self.model))}')
        # while True:
        #     continue
        self.optimizer = self.model.base.optimizer
        # self.optimizer = torch.optim.Adam(
        #     params=trainable_params(self.model),
        #     lr=self.args.local_lr,
        #     weight_decay=self.args.weight_decay,
        # )
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())
        # self.model = torch.nn.DataParallel(self.model)

        self.image_dir = os.path.join(self.model.image_dir, f'trainer_id{self.trainer_id}')
        os.makedirs(self.image_dir, exist_ok=True)

        label_dist = {}

        for k in range(10):
            dist = np.zeros(10)
            # for _k, _v in v['y'].items():
            #     dist[int(_k)] = _v
            # dist = dist / dist.sum()
            # dist[k - 1] = 0.5
            # dist[k] = 0.5
            dist[0] = 1
            label_dist[k] = dist
        # print(f'dist 0: {label_dist[0]}')
        self.label_dist = label_dist
        # while True:
        #     continue
        if trainer_ckpt_name is not None:
            print(f'[Trainer {self.trainer_id}]: loading ckpt {trainer_ckpt_name}')
            self.load_trainer(trainer_ckpt_name)


    def state_dict(self):
        return {
            "trainer_id": self.trainer_id,
            "personal_params_dict": deepcopy(self.personal_params_dict),
            "opt_state_dict": deepcopy(self.opt_state_dict)
        }
        
    def load_state_dict(self, d):
        assert self.trainer_id == d["trainer_id"], f'{self.trainer_id} vs {d["trainer_id"]}'
        self.personal_params_dict = {k: {k1: v1.clone() for k1, v1 in v.items()} for k, v in d["personal_params_dict"].items()}
        self.opt_state_dict = d["opt_state_dict"]
        
    def predefine_dataset(self):
        np.random.seed(0)
        # print(f'num clients: {len(self.data_indices)}')
        # print(f'num items: {[len(v["train"]) for v in self.data_indices]}')
        # print(f'num test items: {[len(v["test"]) for v in self.data_indices]}')
        train_idc = {}
        test_idc = {}
        for i, item in enumerate(self.data_indices):
            train_datas = np.array(item["train"])
            test_datas = np.array(item["test"])
            n_pdf_train = len(train_datas) // 10
            n_pdf_test = len(test_datas) // 10
            trainidc = np.sort(np.random.choice(train_datas, n_pdf_train, replace=False))
            testidc = np.sort(np.random.choice(test_datas, n_pdf_test, replace=False))
            train_idc[i] = trainidc
            test_idc[i] = testidc
        self.train_idc = train_idc
        self.test_idc = test_idc
        # print(f'len train idc: {[len(self.train_idc[i]) for i in range(9)]}')
        # print(f'len test idc: {[len(self.test_idc[i]) for i in range(9)]}')
        # while True:
        #     continue

    # def load_train_dataset(self, e):
    #     """This function is for loading data indices for No.`self.client_id` client."""
    #     # self.trainset.indices = self.data_indices[self.client_id]["train"]
    #     # self.testset.indices = self.data_indices[self.client_id]["test"]
    #     LI = 100
    #     tridc = self.train_idc[self.client_id]
    #     tstidc = self.test_idc[self.client_id]
    #     train_stt = (LI * e) % len(tridc)
    #     test_stt = (LI * e) % len(tstidc)
    #     self.trainset.indices = np.concatenate([tridc, tridc, tridc])[train_stt:train_stt+LI]
    #     self.testset.indices = np.concatenate([tstidc, tstidc, tstidc])[test_stt:test_stt+LI]
        
    #     # self.trainset.indices = self.train_idc[self.client_id]
    #     # self.testset.indices = self.test_idc[self.client_id]
    #     self.trainloader = DataLoader(self.trainset, self.args.batch_size)
    #     if self.args.global_testset:
    #         self.testloader = DataLoader(self.global_testset, self.args.batch_size)
    #     else:
    #         self.testloader = DataLoader(self.testset, self.args.batch_size)
            
    def load_train_dataset(self, e):
        """This function is for loading data indices for No.`self.client_id` client."""
        all_indices = np.concatenate([self.data_indices[self.client_id]["train"], self.data_indices[self.client_id]["test"]])
        self.trainset.indices = all_indices[:math.floor(len(all_indices) * 0.9)]
        self.testset.indices = all_indices[math.floor(len(all_indices) * 0.9):]
        # self.trainset.indices = all_indices[:-2000]
        # self.testset.indices = all_indices[-2000:]
          
        full_size = len(self.trainset.indices)
        print(f'full_size: {full_size}')
        L = full_size * 20
        st = (L * e) % len(self.trainset.indices)
        self.trainset.indices = np.concatenate([self.trainset.indices] * 21)[st: st + L]
    
        # self.trainset.indices = self.train_idc[self.client_id]
        # self.testset.indices = self.test_idc[self.client_id]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size)
        if self.args.global_testset:
            self.testloader = DataLoader(self.global_testset, self.args.batch_size)
        else:
            self.testloader = DataLoader(self.testset, self.args.batch_size)
            
    def load_dataset(self):
        """This function is for loading data indices for No.`self.client_id` client."""
        all_indices = np.concatenate([self.data_indices[self.client_id]["train"], self.data_indices[self.client_id]["test"]])
        self.trainset.indices = all_indices[:math.floor(len(all_indices) * 0.9)]
        self.testset.indices = all_indices[math.floor(len(all_indices) * 0.9):]
        # self.trainset.indices = all_indices[:-1000]
        # self.testset.indices = all_indices[-1000:]
    
    
        # self.trainset.indices = self.train_idc[self.client_id]
        # self.testset.indices = self.test_idc[self.client_id]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size)
        if self.args.global_testset:
            self.testloader = DataLoader(self.global_testset, self.args.batch_size)
        else:
            self.testloader = DataLoader(self.testset, self.args.batch_size)

    def train_and_log(self, verbose=False, eval=False) -> Dict[str, Dict[str, float]]:
        """This function includes the local training and logging process.

        Args:
            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: The logging info, which contains metric stats.
        """
        eval=False
        before = {
            "train_loss": 0,
            "test_loss": 0,
            "train_correct": 0,
            "test_correct": 0,
            "train_size": 1,
            "test_size": 1,
        }
        after = deepcopy(before)
        if eval:
            before = self.evaluate()
        # print(f'before: {before}')
        if self.local_epoch > 0:
            loss_log = self.fit()
            self.save_state()
            # print(f'trained')
            # while True:
            #     continue
            if eval:
                after = self.evaluate()
            # print(f'after: {after}')
        else:
            loss_log = 0
        log_string = ""
        if verbose:
            if len(self.trainset) > 0 and self.args.eval_train:
                log_string = "client [{}] (train)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["train_loss"] / before["train_size"],
                        after["train_loss"] / after["train_size"],
                        before["train_correct"] / before["train_size"] * 100.0,
                        after["train_correct"] / after["train_size"] * 100.0,
                    )
                # self.logger.log(
                #     "client [{}] (train)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                #         self.client_id,
                #         before["train_loss"] / before["train_size"],
                #         after["train_loss"] / after["train_size"],
                #         before["train_correct"] / before["train_size"] * 100.0,
                #         after["train_correct"] / after["train_size"] * 100.0,
                #     )
                # )
            if len(self.testset) > 0 and self.args.eval_test:
                log_string =  "client [{}] (test)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["test_loss"],
                        after["test_loss"],
                        before["test_correct"] / before["test_size"] * 100.0,
                        after["test_correct"] / after["test_size"] * 100.0,
                    )
                # self.logger.log(
                #     "client [{}] (test)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                #         self.client_id,
                #         before["test_loss"] / before["test_size"],
                #         after["test_loss"] / after["test_size"],
                #         before["test_correct"] / before["test_size"] * 100.0,
                #         after["test_correct"] / after["test_size"] * 100.0,
                #     )
                # )

        eval_stats = {"before": before, "after": after, "log_string": log_string, "loss_log": loss_log}
        return eval_stats

    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        """Load model parameters received from the server.

        Args:
            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.
        """
        personal_parameters = self.personal_params_dict.get(
            self.client_id, self.init_personal_params_dict
        )
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )
        # print(f'new params key: {new_parameters.keys()}')
        # print(f'per params key: {personal_parameters.keys()}')
        # print(f'new opt key: {self.opt_state_dict.get(self.client_id, self.init_opt_state_dict).keys()}')
    
        # while True:
        #     continue
        self.model.load_state_dict(new_parameters, strict=False)
        # personal params would overlap the dummy params from new_parameters from the same layerss
        self.model.load_state_dict(personal_parameters, strict=False)

        # if self.client_id not in self.personal_params_dict:
        #     w = self.model.base.model.private_context_generator.codebook.weight
        #     w0 = torch.tensor(np.load(f'/home/server32/minyeong_workspace/FL-bench/data/cifar10_niid3/raw/vq_centroid_client{self.client_id}.npy'))
        #     with torch.no_grad():
        #         w.copy_(w0)
            
        # print(f'w type: {type(w)}')
        # while True:
        #     continue
        # torch.tensor(np.load(f'/home/server32/minyeong_workspace/FL-bench/data/cifar10_niid3/raw/vq_centroid_client{self.client_id}.npy')).to(self.device)
        # if self.client_id in self.all_params_dict:
        #     prev = self.all_params_dict.get(self.client_id, self.init_personal_params_dict)
        #     print(f'prev: {list(prev.items())[:10]}')
        #     now = self.model.state_dict()
        #     dv = {}
        #     for k in prev.keys():
        #         nv = now[k].clone().detach().cpu()
        #         pv = prev[k]
        #         try:
        #             delta_v = (nv - pv).norm()
        #             dv[k] = delta_v
        #         except:
        #             continue
        #     print(f'dv: {dv}')
        #     while True:
        #         continue
        
    def optimizer_to(self, device):
        optim = self.optimizer
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def save_state(self):
        """Save client model personal parameters and the state of optimizer at the end of local training."""
        self.personal_params_dict[self.client_id] = {
            key: param.clone().detach().cpu()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (key in self.personal_params_name)
        }
        # print(f'opt state dict value types: {[(k, type(v)) for k, v in self.optimizer.state_dict()["state"][0].items()]} ')
        
        # 31it [00:00, 49.00it/s]opt state dict value types: [('step', <class 'torch.Tensor'>), ('exp_avg', <class 'torch.Tensor'>), ('exp_avg_sq', <class 'torch.Tensor'>)] 
        # while True:
        #     continue
        # self.optimizer_to('cpu')
        # self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())
        # self.optimizer_to(self.device)
        # self.all_params_dict[self.client_id] = {
        #     key: param.clone().detach().cpu()
        #     for key, param in self.model.state_dict(keep_vars=True).items()
        # }
        
    def load_trainer(self, name):
        ppd_path = os.path.join(name + '_after.pt')
        opt_path = os.path.join(name + '_after_opt.pt')
        self.load_ppd(ppd_path)
        if os.path.exists(opt_path):
            self.load_opt(opt_path)
                                
    def load_ppd(self, path):
        d = torch.load(path)
        self.personal_params_dict.update(d)
        # for name, v in cid_ppd.items():
        #     self.personal_params_dict[cid][name] = v
    def load_opt(self, path):
        d = torch.load(path)
        self.opt_state_dict.update(d)
        
    def save_trainer(self, epoch, out_dir, before=False):
        postfix = 'after' if not before else 'before'
        model_name = (
            f"{self.args.dataset}_{epoch}_{self.args.model}_trainer{self.trainer_id}_{postfix}.pt"
        )
        opt_name = (
            f"{self.args.dataset}_{epoch}_{self.args.model}_trainer{self.trainer_id}_{postfix}_opt.pt"
        )
        ppd_to_save = {}
        ppd_to_save_keys = trainable_params(self.model, requires_name=True)[1]
        
        # print(f'ps params dict: {self.init_personal_params_dict.keys()}')
        # while True:
        #     continue
        for cid in self.personal_params_dict.keys():
            ppdcid = self.personal_params_dict[cid]
            ppdcid_ts = {k: ppdcid[k] for k in ppd_to_save_keys if k in ppdcid}
            ppd_to_save[cid] = ppdcid_ts
            # print(f'ppdcid_ts keys: {ppdcid_}')
        torch.save(ppd_to_save, out_dir / model_name)
        torch.save(self.opt_state_dict, out_dir / opt_name)

    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        return_diff=True,
        verbose=False,
        epoch=None,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
        """
        The funtion for including all operations in client local training phase.
        If you wanna implement your method, consider to override this funciton.

        Args:
            client_id (int): The ID of client.

            local_epoch (int): The number of epochs for performing local training.

            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.

            return_diff (bool, optional):
            Set as `True` to send the difference between FL model parameters that before and after training;
            Set as `False` to send FL model parameters without any change.  Defaults to True.

            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
            [The difference / all trainable parameters, the weight of this client, the evaluation metric stats].
        """
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_train_dataset(epoch)
        self.set_parameters(new_parameters)
        # print(f'new_params: {list(new_parameters.items())[:10]}')
        eval_stats = self.train_and_log(verbose=verbose, eval=((epoch + 1) % 20 == 0))
 
        client_params =  {name: param.detach().cpu() for name, param in self.model.state_dict(keep_vars=True).items()}
        shared_parameters = [client_params[k] for k in new_parameters]

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), shared_parameters
            ):
                delta[name] = p0 - p1

            return delta, len(self.trainset), eval_stats
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_stats,
            )

    def fit(self):
        """
        The function for specifying operations in local training phase.
        If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
        """
        self.model.train()
        # scaler = torch.cuda.amp.GradScaler()
        cnt = 0
        loss_log = 0
        for _ in range(self.local_epoch):
            loss = 0
            print(f'len trainloadr: {len(self.trainloader)}')
            for j, (x, y) in tqdm(enumerate(self.trainloader)):
                # y.fill_(self.client_id)
                # print(f'y: {y}')
                # while True:
                #     continue
                loss = self.model.step(x, y)
                loss_log += loss * len(y)
                cnt += len(y)
                print(f'[trainer{self.trainer_id}] train loss: {loss}')
                # cnt += 1
                # # inst_batch = {
                # #     "image": inst_x,
                # #     "caption": list(inst_c)
                # # }
                # # print(f'x type: {type(x)}')
                # # print(f'y type: {type(y)}')
                # # print(f'x shape: {x.shape}')
                # # print(f'y shape: {y.shape}')
                # # print(f'x0: {x[0]}')
                # # print(f'mean x: {x.mean()}, var x: {x.var()}')
                # # print(f'y: {y}')
                # # while True:
                # #     continue
                # # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # # So the latent size 1 data batches are discarded.
                # # if len(x) <= 1:
                # #     continue
                
                # # print(f'inst_x shpape: {x.shape}')
                # # print(f'inst_c: {inst_c}')
                # # while True:
                # #     continue
                # x, y = x.to(self.device), y.to(self.device)
                # if cnt == 1:
                #     self.optimizer.zero_grad()
                    
                # # print(f'cnt : {cnt}')
                # # with torch.autocast(device_type='cuda', dtype=torch.float16):
                #     # res = self.model(x, inst_batch)
                #     # loss += self.model.compute_loss(res)
                #     # print(f'loss type: {loss.type()}')
                #     # while True:
                #     #     continue
                # res = self.model(x, y)
                # loss += self.model.compute_loss(res)
                    
                # if (cnt == 1) or (j == len(self.trainloader) - 1):
                #     # print(f'loss: {loss}')
                #     # scaler.scale(loss / cnt).backward()
                #     # scaler.step(self.optimizer)
                #     # scaler.update()
                #     loss.backward()
                #     self.optimizer.step()
                    
                #     cnt = 0
                #     loss = 0

                # assert cnt < 1, f'cnt: {cnt}'
        
        # print(f'train cnt: {cnt}')
        # while True:
        #     continue
        loss_log = loss_log / cnt  
        return loss_log
    
    @torch.no_grad()
    def evaluate(
        self, model: torch.nn.Module = None, test_flag=False
    ) -> Dict[str, float]:
        """The evaluation function. Would be activated before and after local training if `eval_test = True` or `eval_train = True`.

        Args:
            model (torch.nn.Module, optional): The target model needed evaluation (set to `None` for using `self.model`). Defaults to None.
            test_flag (bool, optional): Set as `True` when the server asking client to test model.
        Returns:
            Dict[str, float]: The evaluation metric stats.
        """
        # disable train data transform while evaluating
        self.dataset.enable_train_transform = False

        eval_model = self.model if model is None else model
        eval_model.eval()
        train_loss, test_loss = 0, 0
        train_correct, test_correct = 0, 0
        train_sample_num, test_sample_num = 0, 0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.testset) > 0 and (test_flag or self.args.eval_test):
            test_loss, test_correct, test_sample_num = evalutate_model(
                model=eval_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.eval_train:
            train_loss, train_correct, train_sample_num = evalutate_model(
                model=eval_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )

        self.dataset.enable_train_transform = True

        # print(f'cid{self.client_id} - test samples: {test_sample_num}')
        # while True:
        #     continue
        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_correct": train_correct,
            "test_correct": test_correct,
            "train_size": float(max(1, train_sample_num)),
            "test_size": float(max(1, test_sample_num)),
        }

    def calc_fid(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor], epoch: int, save_dir
    ) -> Dict[str, Dict[str, float]]:
        """Test function. Only be activated while in FL test round.

        Args:
            client_id (int): The ID of client.
            new_parameters (OrderedDict[str, torch.Tensor]): The FL model parameters.

        Returns:
            Dict[str, Dict[str, float]]: the evalutaion metrics stats.
        """
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        before = {
            "train_loss": 0,
            "train_correct": 0,
            "train_size": 1.0,
            "test_loss": 0,
            "test_correct": 0,
            "test_size": 1.0,
        }
        after = deepcopy(before)

        before = self.evaluate(test_flag=True)
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate(test_flag=True)
            
         
        print(f'evaluating')
        self.model.base.model.eval()
        local_save_path = save_dir / f'{epoch}' / 'local' / f'{self.client_id}'
        global_save_path = save_dir / f'{epoch}' / 'global' / f'{self.client_id}'
        adv_save_path =save_dir / f'{epoch}' / 'adv' / f'{self.client_id}'
        os.makedirs(local_save_path, exist_ok=True)
        os.makedirs(global_save_path, exist_ok=True)
        os.makedirs(adv_save_path, exist_ok=True)
        # fid_adv = self.model.evaluator.sample(self.model.base.sample_fn, self.label_dist[self.client_id], [f'/home/server32/minyeong_workspace/FL-bench/data/pathmnist/ref_fid_stats_pathmnist_client{self.client_id}.npz'], is_leader=True, adv=True, save_path=adv_save_path)['fid'][0]
        fid_local = self.model.evaluator.sample(self.model.base.sample_fn, self.label_dist[self.client_id], [f'/home/server32/minyeong_workspace/FL-bench/data/pathmnist/ref_fid_stats_pathmnist_client{self.client_id}.npz'], is_leader=True, save_path=local_save_path)['fid'][0]
        # fid_global = self.model.evaluator.sample(self.model.base.sample_fn, self.label_dist[self.client_id], [f'/home/server32/minyeong_workspace/FL-bench/data/pathmnist/ref_fid_stats_pathmnist.npz'], is_leader=True, save_path=global_save_path)['fid'][0]
        fid_adv = fid_local
        fid_global = fid_local
        
        
        eval_results = [fid_local, fid_global, fid_adv]
        print(f'[Trainer {self.trainer_id}] evaluated: {eval_results}')
        # while True:
        #     continue


        #     if (not (e + 1) % self.image_intv or self.dry_run) and self.num_samples and image_dir:
        #         self.model.eval()
        #         x = self.sample_fn(diffusion=self.diffusion_eval, sample_size=self.num_samples, sample_seed=self.sample_seed).cpu()
        #         if self.is_leader:
        #             save_image(x, os.path.join(image_dir, f"{e + 1}.jpg"), nrow=nrow)

        #     if not (e + 1) % self.chkpt_intv and chkpt_path:
        #         self.model.eval()
        #         if evaluator is not None:
        #             eval_results = evaluator.eval(self.sample_fn, is_leader=self.is_leader)
        #         else:
        #             eval_results = dict()
        #         results.update(eval_results)
        #         if self.is_leader:
        #             self.save_checkpoint(chkpt_path, epoch=e + 1, **results)
                    
        
        return {"before": before, "after": after, 'fid_global': eval_results[1], 'fid_local': eval_results[0], 'fid_adv': eval_results[2]}


    def test(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor], epoch: int
    ) -> Dict[str, Dict[str, float]]:
        """Test function. Only be activated while in FL test round.

        Args:
            client_id (int): The ID of client.
            new_parameters (OrderedDict[str, torch.Tensor]): The FL model parameters.

        Returns:
            Dict[str, Dict[str, float]]: the evalutaion metrics stats.
        """
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        before = {
            "train_loss": 0,
            "train_correct": 0,
            "train_size": 1.0,
            "test_loss": 0,
            "test_correct": 0,
            "test_size": 1.0,
        }
        after = deepcopy(before)

        before = self.evaluate(test_flag=True)
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate(test_flag=True)
            
        
        print(f'evaluating')
        self.model.base.model.eval()
        save_path = os.path.join(self.image_dir, f'{self.client_id}_{epoch}.png')
        fid_adv = self.model.evaluator.eval(self.model.base.sample_fn, self.label_dist[self.client_id], [f'/home/server32/minyeong_workspace/FL-bench/data/pathmnist/fid_stats_pathmnist_client{self.client_id}.npz'], is_leader=True, adv=True)['fid'][0]
        fid_local = self.model.evaluator.eval(self.model.base.sample_fn, self.label_dist[self.client_id], [f'/home/server32/minyeong_workspace/FL-bench/data/pathmnist/fid_stats_pathmnist_client{self.client_id}.npz'], is_leader=True, save_path=save_path)['fid'][0]
        fid_global = self.model.evaluator.eval(self.model.base.sample_fn, np.ones_like(self.label_dist[self.client_id]) / len(self.label_dist[self.client_id]), [f'/home/server32/minyeong_workspace/FL-bench/data/pathmnist/fid_stats_pathmnist.npz'], is_leader=True)['fid'][0]

        eval_results = [fid_local, fid_global, fid_adv]
        print(f'evaluated: {eval_results}')
        # while True:
        #     continue


        #     if (not (e + 1) % self.image_intv or self.dry_run) and self.num_samples and image_dir:
        #         self.model.eval()
        #         x = self.sample_fn(diffusion=self.diffusion_eval, sample_size=self.num_samples, sample_seed=self.sample_seed).cpu()
        #         if self.is_leader:
        #             save_image(x, os.path.join(image_dir, f"{e + 1}.jpg"), nrow=nrow)

        #     if not (e + 1) % self.chkpt_intv and chkpt_path:
        #         self.model.eval()
        #         if evaluator is not None:
        #             eval_results = evaluator.eval(self.sample_fn, is_leader=self.is_leader)
        #         else:
        #             eval_results = dict()
        #         results.update(eval_results)
        #         if self.is_leader:
        #             self.save_checkpoint(chkpt_path, epoch=e + 1, **results)
                    
        
        return {"before": before, "after": after, 'fid_global': eval_results[1], 'fid_local': eval_results[0], 'fid_adv': eval_results[2]}


    def finetune(self):
        """
        The fine-tune function. If your method has different fine-tuning opeation, consider to override this.
        This function will only be activated while in FL test round.
        """
        self.model.train()
        for _ in range(self.args.finetune_epoch):
            for x, y, inst_x, inst_c in self.trainloader:
                inst_batch = {
                    "image": inst_x,
                    "caption": list(inst_c)
                }
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x, inst_batch)
                loss = self.model.compute_loss(out)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
