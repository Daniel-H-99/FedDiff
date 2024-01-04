import os
import pickle as pkl
import wandb

EPOCHS = [500, 1000, 1500, 2000, 2500, 3000]
NUM_CLI = 8
TAG = "fed"

def init_wandb():
    wandb.init(project='report', name=f'fid_{TAG}')
    
def avg(values):
    assert type(values) == list, f'type of {type(values)}'
    assert len(values) > 0, f'length of {len(values)}'
    return sum(values) / len(values)


def main():
    fed = {}
    with open(f'tested_fid_{TAG}_client_0.pkl', 'rb') as f:
        fed[0] = pkl.load(f)

    with open(f'tested_fid_{TAG}_client_1.pkl', 'rb') as f:
        fed[1] = pkl.load(f)     
        
    with open(f'tested_fid_{TAG}_client_2.pkl', 'rb') as f:
        fed[2] = pkl.load(f)  

    with open(f'tested_fid_{TAG}_client_3.pkl', 'rb') as f:
        fed[3] = pkl.load(f)  

    with open(f'tested_fid_{TAG}_client_4.pkl', 'rb') as f:
        fed[4] = pkl.load(f)  
        
    with open(f'tested_fid_{TAG}_client_5.pkl', 'rb') as f:
        fed[5] = pkl.load(f)  
        
    with open(f'tested_fid_{TAG}_client_6.pkl', 'rb') as f:
        fed[6] = pkl.load(f)  

    with open(f'tested_fid_{TAG}_client_7.pkl', 'rb') as f:
        fed[7] = pkl.load(f)  
        

        
              
    # cli = {}
    # with open('tested_fid_client_0.pkl', 'rb') as f:
    #     cli[0] = pkl.load(f)

    # with open('tested_fid_client_1.pkl', 'rb') as f:
    #     cli[1] = pkl.load(f)     
        
    # with open('tested_fid_client_2.pkl', 'rb') as f:
    #     cli[2] = pkl.load(f)  

    # with open('tested_fid_client_3.pkl', 'rb') as f:
    #     cli[3] = pkl.load(f)  

    # with open('tested_fid_client_4.pkl', 'rb') as f:
    #     cli[4] = pkl.load(f)  
        
    # with open('tested_fid_client_5.pkl', 'rb') as f:
    #     cli[5] = pkl.load(f)  
        
    # with open('tested_fid_client_6.pkl', 'rb') as f:
    #     cli[6] = pkl.load(f)  

    # with open('tested_fid_client_7.pkl', 'rb') as f:
    #     cli[7] = pkl.load(f)  
             
    init_wandb()

    for e in EPOCHS:
        print(f'working on epoch {e}')
        local_local_list = []
        local_global_list = []
        global_global_list = []
        
        for cid in range(NUM_CLI):
            print(f'working on client {cid}')
            local_local = fed[cid][e][f'local_local_client_{cid}']
            local_global = fed[cid][e][f'local_global_client_{cid}']
            global_global = fed[cid][e][f'global_global_client_{cid}']
            # cli_local_local = cli[cid][e][f'local_local_client_{cid}']
            # cli_local_global = cli[cid][e][f'local_global_client_{cid}']
            # cli_global_global = cli[cid][e][f'global_global_client_{cid}']
            wandb.log({
                f"local_local_client_{cid}": local_local,
                f"local_global_client_{cid}": local_global,
                f"global_global_client_{cid}": global_global,
            }, step=e)
            
            local_local_list.append(local_local)
            local_global_list.append(local_global)
            global_global_list.append(global_global)
            # local_local_list.append(fed_local_local)
            # local_global_list.append(fed_local_global)
            # global_global_list.append(fed_global_global)
            
        local_local_avg = avg(local_local_list)
        local_global_avg = avg(local_global_list)
        global_global_avg = avg(global_global_list)
        # local_local_avg = avg(fed_local_local_list)
        # local_global_avg = avg(fed_local_global_list)
        # global_global_avg = avg(fed_global_global_list)
        
        wandb.log({
            f"local_local_avg": local_local_avg,
            f"local_global_avg": local_global_avg,
            f"global_global_avg": global_global_avg,
        }, step=e)
        
        # wandb.log({
        #     f"local_local_avg": local_local_avg,
        #     f"local_global_avg": local_global_avg,
        #     f"global_global_avg": global_global_avg,
        # }, step=e)
            
            
if __name__=="__main__":
    main()