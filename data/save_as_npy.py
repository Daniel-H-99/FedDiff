import numpy as np
import os
from torchvision.datasets import MNIST
from tqdm import tqdm
from imageio import imwrite

def main():
    root = '.'
    save_dir = 'MNIST/raw'
    os.makedirs(save_dir, exist_ok=True)
    d_train = MNIST(root=root, train=True, download=True)
    d_test = MNIST(root=root, train=False, download=True)
    xdata = []
    ydata = []
    
    for d in [d_train, d_test]:
        for item in tqdm(d):
            x = item[0]
            y = item[1]
            
            ### resize to 32 ###
            x = x.resize((32, 32))
            x = np.array(x)
            ### to rgb ###
            x = np.repeat(x[:, :, None], 3, axis=2)
            
            xdata.append(x)
            ydata.append(y)
    
    
    xdata = np.stack(xdata)
    ydata = np.array(ydata)
    
    np.save(os.path.join(save_dir, 'xdata.npy'), xdata)
    np.save(os.path.join(save_dir, 'ydata.npy'), ydata)
    
    
if __name__=='__main__':
    main()
        
        