import os
import yaml
import argparse
from DiffGAD import DiffGAD

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', dest='device', type=int, default=0)
    parser.add_argument('--config', dest='config', type=str, default='configs/books.yaml')
    args = parser.parse_args()
    return args

args = get_arguments()
cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

dset = cfg['dataset']
if dset in ['books', 'disney', 'reddit', 'weibo', 'enron']:
    DiffGAD = DiffGAD(lr=0.004, ae_alpha=cfg['ae_alpha'], ae_lr=cfg['ae_lr'], ae_dropout=cfg['ae_dropout'], proto_alpha = cfg['proto_alpha'], hid_dim = cfg['hid_dim'], weight=cfg['weight'])
    print("ae_alpha: {:.4f}, ae_lr: {:.4f}, ae_dropout: {:.4f}".format(cfg['ae_alpha'], cfg['ae_lr'], cfg['ae_dropout']))

DiffGAD(dset)