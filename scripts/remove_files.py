import os 
import shutil

def main():
    path = '/home/server33/minyeong_workspace/FL-bench/out_cifar10_niid2_phoenix_trial1/FedDiff/checkpoints'
    files = os.listdir(path)
    epoch = lambda x: int(x.split('_')[2])
    for f in files:
        if epoch(f) < 20 and 'opt' in f:
            os.remove(os.path.join(path, f))



if __name__=='__main__':
    main()