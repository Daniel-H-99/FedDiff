import importlib
import sys
import inspect
from pathlib import Path
import wandb
sys.path.append(Path(__file__).parent.joinpath("src/server").absolute().as_posix())


def wandb_init():
    wandb.init(project="pathvae", name="cifar10_class0_fed_trial1")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Need to assign a method. Run like `python main.py <method> [args ...]`, e.g., python main.py fedavg -d cifar10 -m lenet5`"
        )
    wandb_init()
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

    server = server_class(args=parser.parse_args(args_list))

    server.run()
