import torch
import argparse
import os

def main(path, idx, channels):
    pred = []
    for ori in ['xy', 'yz', 'zx']:
        pred.append(torch.load(os.path.join(path, f'pred_{idx:03d}_{ori}_2d.pt')))
        print(pred[-1].shape)

    voted = sum(pred) >= 2
    discard = sum(pred) == 1
    print(discard.sum()/voted.sum())
    torch.save(voted, os.path.join(path, f'pred_{idx:03d}_voted_2d'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='output')
    parser.add_argument('--idx', type=int, default=7)
    parser.add_argument('--channels', type=int, default=5)
    args = parser.parse_args()
    main(args.path, args.idx, args.channels)