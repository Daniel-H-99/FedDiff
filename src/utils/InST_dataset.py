import pickle
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

from src.utils.tools import trainable_params, evalutate_model, Logger
from src.utils.models import DecoupledModel
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS
import numpy as np

class FedDiffClient:
    def __init__(
        self,
        model: DecoupledModel,
        args: Namespace,
        logger: Logger,
        device: torch.device,
    ):
        self.args = args
        self.device = device
        self.client_id: int = None

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
            [transforms.Resize((32, 32)), transforms.Normalize((np.array(MEAN[self.args.dataset]) * 255).tolist(), (np.array(STD[self.args.dataset]) * 255).tolist())]
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

        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger = logger
        self.personal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.personal_params_name: List[str] = []
        self.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        self.opt_state_dict = {}

        self.optimizer = torch.optim.SGD(
            params=trainable_params(self.model),
            lr=self.args.local_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())

    def load_dataset(self):
        """This function is for loading data indices for No.`self.client_id` client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.testset.indices = self.data_indices[self.client_id]["test"]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size)
        if self.args.global_testset:
            self.testloader = DataLoader(self.global_testset, self.args.batch_size)
        else:
            self.testloader = DataLoader(self.testset, self.args.batch_size)