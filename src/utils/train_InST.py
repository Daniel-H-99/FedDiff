
from InST.export import export
from InST_dataset import PathMNIST
from torch.utils.data import DataLoader
import wandb

def init_wandb():
    wandb.init({"project": "test_InST", "name": "test"})
    
def main():
    ### setup ###
    model, optimizer = load_model_optimzer()
    dataset = PathMNIST()

    batch_size = 1
    dl_train = DataLoader(dataset["train"], batch_size=batch_size)
    dl_val = DataLoader(dataset["val"], batch_size=batch_size)
    
    
    init_wandb()
    
    
    
    


