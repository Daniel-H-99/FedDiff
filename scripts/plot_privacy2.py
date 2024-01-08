import os
import pickle as pkl
import wandb
import numpy as np

def init_wandb():
    wandb.init(project='report', name='privacy2_fed_class0')
    
def avg(values):
    assert type(values) == list, f'type of {type(values)}'
    assert len(values) > 0, f'length of {len(values)}'
    return sum(values) / len(values)

def np_avg(values):
    assert type(values) == list, f'type of {type(values)}'
    assert len(values) > 0, f'length of {len(values)}'
    flattened = np.concatenate(values)
    return flattened.mean()

EPOCHS = [400]
NUM_CLI = 1
TAG="fed"
def main():
    fed = {}
    with open(f'tested_privacy2_{TAG}_client_0.pkl', 'rb') as f:
        fed[0] = pkl.load(f)

    # with open(f'tested_privacy2_{TAG}_client_1.pkl', 'rb') as f:
    #     fed[1] = pkl.load(f)     
        
    # with open(f'tested_privacy2_{TAG}_client_2.pkl', 'rb') as f:
    #     fed[2] = pkl.load(f)  

    # with open(f'tested_privacy2_{TAG}_client_3.pkl', 'rb') as f:
    #     fed[3] = pkl.load(f)  

    # with open(f'tested_privacy2_{TAG}_client_4.pkl', 'rb') as f:
    #     fed[4] = pkl.load(f)  
        
    # with open(f'tested_privacy2_{TAG}_client_5.pkl', 'rb') as f:
    #     fed[5] = pkl.load(f)  
        
    # with open(f'tested_privacy2_{TAG}_client_6.pkl', 'rb') as f:
    #     fed[6] = pkl.load(f)  

    # with open(f'tested_privacy2_{TAG}_client_7.pkl', 'rb') as f:
    #     fed[7] = pkl.load(f)  
        


    init_wandb()

    for e in EPOCHS:
        print(f'working on epoch {e}')
        local_train_list = []
        local_test_list = []
        global_train_list = []
        global_test_list = []
        local_gap_list = []
        global_gap_list = []
        local_ratio_list = []
        global_ratio_list = []
        for cid in range(NUM_CLI):
            print(f'working on client {cid}')
            # local_train = fed[cid][e][f'local_train_client_{cid}']
            # local_test = fed[cid][e][f'local_test_client_{cid}']
            global_train = fed[cid][e][f'global_train_client_{cid}']
            global_test = fed[cid][e][f'global_test_client_{cid}']
            # local_ratio = local_test / local_train.clip(min=1e-6)
            global_ratio = global_test / global_train.clip(min=1e-6)

            wandb.log({
                # f"local_train_client_{cid}": local_train,
                # f"local_test_client_{cid}": local_test,
                f"global_train_client_{cid}": global_train,
                f"global_test_client_{cid}": global_test,
                # f"local_gap_client_{cid}": local_test - local_train,
                f"global_gap_client_{cid}": global_test - global_train,
                # f'local_ratio_client_{cid}': local_ratio,
                f'global_ratio_client_{cid}': global_ratio,
                # f'local_ratio_avg_client_{cid}': local_ratio.mean(axis=0),
                f'global_ratio_avg_client_{cid}': global_ratio.mean(axis=0),
            }, step=e)
            
            # local_train_list.append(local_train)
            # local_test_list.append(local_test)
            global_train_list.append(global_train)
            global_test_list.append(global_test)
            # local_gap_list.append(local_test - local_train)
            global_gap_list.append(global_test - global_train)
            # local_ratio_list.append(local_ratio)
            global_ratio_list.append(global_ratio)
            
        # local_train_avg = avg(local_train_list)
        # local_test_avg = avg(local_test_list)
        global_train_avg = avg(global_train_list)
        global_test_avg = avg(global_test_list)
        # local_gap_avg = avg(local_gap_list)
        global_gap_avg = avg(global_gap_list)
        # local_ratio_all = np.concatenate(local_gap_list)
        global_ratio_all = np.concatenate(global_gap_list)
        # local_ratio_avg = np_avg(local_ratio_list)
        global_ratio_avg = np_avg(global_ratio_list)
        
        
        wandb.log({
            # f"local_train_avg": local_train_avg,
            # f"local_test_avg": local_test_avg,
            f"global_train_avg": global_train_avg,
            f"global_test_avg": global_test_avg,
            # f"local_gap_avg": local_gap_avg,
            f"global_gap_avg": global_gap_avg,
            # f"local_ratio_all": local_ratio_all,
            f"global_ratio_all": global_ratio_all,
            # f"local_ratio_avg": local_ratio_avg,
            f"global_ratio_avg": global_ratio_avg,
        }, step=e)
        
            
if __name__=="__main__":
    main()