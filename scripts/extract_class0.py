import os
import pickle as pkl


def main():
    data_dir = '/home/server33/minyeong_workspace/FL-bench/data/cifar10_class0'
    save_path = '/home/server33/minyeong_workspace/FL-bench/data/cifar10_class0/parition.pkl'
    indices_path = '/home/server33/minyeong_workspace/FL-bench/data/cifar10_class0/indices.pkl'
    out = {'separation': {}}
    out['separation']['train'] = list(range(7))
    out['separation']['test'] = list(range(7))
    out['separation']['total'] = list(range(7))
    indices = pkl.load(open(indices_path, 'rb'))
    split_indices = [{'train': v[:int(0.9 * len(v))], 'test': v[int(0.9 * len(v)):]} for v in indices.values()]
    out['data_indices'] = split_indices
    with open(save_path, 'wb') as f:
        pkl.dump(out, f)


if __name__=='__main__':
    main()

