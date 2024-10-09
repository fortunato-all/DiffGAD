import argparse
import os
from diffgad import DiffGAD


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', dest='device', type=int, default=5)
    parser.add_argument('--lamda', dest='lamda', type=float, default=0.2)
    parser.add_argument('--dataset', dest='dataset', type=str, default='disney')
    parser.add_argument('--ae_lr', dset='ae_lr', type=float, default=0.05)
    parser.add_argument('--ae_alpha', dset='ae_alpha', type=float, default=0.1)
    parser.add_argument('--ae_dropout', dset='ae_dropout', type=float, default=0.3)
    args = parser.parse_args()
    return args

args = get_arguments()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)     # set GPU
dset = args.dataset
model = DiffGAD(lr=0.004, ae_alpha=args.ae_alpha, ae_lr=args.ae_lr, ae_dropout=args.ae_dropout, lamda=args.lamda)
model(dset)