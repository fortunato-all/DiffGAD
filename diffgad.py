import tqdm
import torch
from pygod.metric import *
from pygod.utils import load_data
from torch_geometric.datasets import DGraphFin

import os
import tqdm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from denoise_model import extract
from diffusion_models import MLPDiffusion, Model, sample_dm, sample_dm_free
from denoise_model import DenoiseNN, p_losses, sample, q_sample, extract
from datetime import datetime

from pygod.metric.metric import *
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj
from pygod.nn.decoder import DotProductDecoder
from pygod.nn.functional import double_recon_loss


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# define beta schedule
betas = linear_beta_schedule(timesteps=500)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

class Graph_AE(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 **kwargs):
        super(Graph_AE, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.shared_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)

        self.attr_decoder = backbone(in_channels=hid_dim,
                                     hidden_channels=hid_dim,
                                     num_layers=decoder_layers,
                                      out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)

        self.struct_decoder = DotProductDecoder(in_dim=hid_dim,
                                                hid_dim=hid_dim,
                                                num_layers=decoder_layers - 1,
                                                dropout=dropout,
                                                act=act,
                                                sigmoid_s=sigmoid_s,
                                                backbone=backbone,
                                                **kwargs)

        self.loss_func = double_recon_loss
        self.emb = None

    def forward(self, x, edge_index):
        self.emb = self.encode(x, edge_index)
        x_, s_ = self.decode(self.emb, edge_index)
        return x_, s_, self.emb
    
    def encode(self, x, edge_index):
        self.emb = self.shared_encoder(x, edge_index)
        return self.emb
    
    def decode(self, emb, edge_index):
        x_ = self.attr_decoder(emb, edge_index)
        s_ = self.struct_decoder(emb, edge_index)
        return x_, s_


class DiffGAD(BaseTransform):
    def __init__(self,
                 name="",
                 hid_dim=None,
                 diff_dim=None,
                 ae_epochs=300,
                 diff_epochs=800,
                 patience=100,
                 lr=0.005,
                 wd=0.,
                 lamda=0.0,
                 sample_steps=50,
                 radius = 1,
                 ae_dropout=0.3,
                 ae_lr=0.05, 
                 ae_alpha=0.8,
                 verbose=True):

        self.name = name
        self.hid_dim = hid_dim
        self.diff_dim = diff_dim
        self.ae_epochs = ae_epochs
        self.diff_epochs = diff_epochs
        self.patience = patience
        self.lr = lr
        self.wd = wd
        self.sample_steps = sample_steps
        self.verbose = verbose
        self.lamda = lamda
        
        self.common_feat = None
        self.dm = None

        self.ae = None
        self.ae_dropout = ae_dropout
        self.ae_lr = ae_lr
        self.ae_alpha = ae_alpha
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.timesteps = 500
        self.radius = radius
        

    def forward(self, dset):
        self.dataset = dset 
        data = load_data(self.dataset)

        if self.hid_dim is None:
            self.hid_dim = 2 ** int(math.log2(data.x.size(1)) - 1)
        if self.diff_dim is None:
            self.diff_dim = 2 * self.hid_dim
        
        self.ae = Graph_AE(in_dim = data.num_node_features, 
                           hid_dim=self.hid_dim,
                           dropout=self.ae_dropout).cuda()
        self.save_dir = os.getcwd() + '/models/'+ self.dataset + '/full_batch/'
        self.ae_path = self.save_dir + "/" + str(self.ae_dropout) + "_" + str(self.ae_lr) + "_" + str(self.ae_alpha) + "_" + str(self.hid_dim)
        if not os.path.exists(self.ae_path):
            os.makedirs(self.ae_path)
        ######################## train autoencoder #######################
        self.train_ae(data)
        ae_dict = torch.load(self.ae_path + '/Graph_AE.pt')
        self.ae.load_state_dict(ae_dict['state_dict'])

        num_trial = 20
        for _ in tqdm.tqdm(range(num_trial)):
            ##################################
            # unconditional diffusion models
            denoise_fn = MLPDiffusion(self.hid_dim, self.diff_dim).cuda()
            self.dm = Model(denoise_fn=denoise_fn,
                            hid_dim=self.hid_dim).cuda()
            self.common_feat = self.train_dm(data)
            dm_dict = torch.load(self.ae_path + '/edm.pt')
            self.dm.load_state_dict(dm_dict['state_dict'])
            self.common_feat = dm_dict['common_feat']
            #################################
            # conditional diffusion models
            print(self.common_feat)
            denoise_condition = MLPDiffusion(self.hid_dim, self.diff_dim).cuda()
            self.dm_conditon = Model(denoise_fn=denoise_condition,
                            hid_dim=self.hid_dim).cuda()
            self.train_dm_conditon(data)
            dm_free_dict = torch.load(self.ae_path + '/conditon_edm.pt')
            self.dm_conditon.load_state_dict(dm_free_dict['state_dict'])
            #################################
            # evaluation
            self.sample_free(self.dm_conditon, self.dm, data)


    def train_ae(self, data):
        if self.verbose:
            print('Training autoencoder ...')
        lr = self.ae_lr
        optimizer = torch.optim.Adam(self.ae.parameters(), lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)   
        
        for epoch in range(1, self.ae_epochs+1):
            self.ae.train()
            optimizer.zero_grad()

            x = data.x.cuda()
            edge_index = data.edge_index.cuda()
            y = data.y.bool()
            s = to_dense_adj(edge_index)[0].cuda()
            x_, s_, embedding = self.ae(x, edge_index)
            score = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)
            loss = torch.mean(score)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            save_path = self.ae_path
            torch.save({
                'state_dict': self.ae.state_dict(),
            }, save_path + '/Graph_AE.pt')


    def train_dm(self, data):
        if self.verbose:
            print('Training diffusion model ...')      
        optimizer = torch.optim.Adam(self.dm.parameters(), lr=self.lr,
                                     weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 
        self.dm.train()
        best_loss = float('inf') 

        best_auc = 0
        patience = 0
        common_feat = None
        for epoch in range(self.diff_epochs):
            x = data.x.cuda()
            edge_index = data.edge_index.cuda()
            inputs = self.ae.encode(x, edge_index) 

            if epoch == 0:
                common_feat = torch.mean(inputs, dim=0)
            else:
                s_v = self.cos(common_feat, reconstructed)
                omega = softmax_with_temperature(s_v,t=5).reshape(1,-1)
                common_feat = torch.mm(omega, reconstructed).detach() 

            loss, score_train, reconstructed = self.dm(inputs)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dm.parameters(), 1.0)
            optimizer.step()
            if epoch%10 == 0:
                print("Epoch:", '%04d' % (epoch), "loss=", "{:.5f}".format(loss))
            scheduler.step()

            if loss < best_loss:
                best_loss = loss
                patience = 0
                save_dir = self.ae_path
                torch.save({
                'state_dict': self.dm.state_dict(),
                'common_feat': common_feat
                }, save_dir + '/edm.pt')
            else:
                patience += 1
                if patience == self.patience:
                    if self.verbose:
                        print('Early stopping')
                    break

        return common_feat
    
    def train_dm_condition(self, data):
        if self.verbose:
            print('Training diffusion model ...')      
        optimizer = torch.optim.Adam(self.dm_condition.parameters(), lr=self.lr,
                                     weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 
        self.dm_condition.train()
        best_loss = float('inf') 

        best_auc = 0
        patience = 0
        for epoch in range(self.diff_epochs):
            x = data.x.cuda()
            edge_index = data.edge_index.cuda()

            inputs = self.ae.encode(x, edge_index)
            loss, score_train, reconstructed = self.dm_condition(inputs, common_feat = self.common_feat)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dm_condition.parameters(), 1.0)
            optimizer.step()
            if epoch%10 == 0:
                print("Epoch:", '%04d' % (epoch), "loss=", "{:.5f}".format(loss))
            scheduler.step()

            if loss < best_loss:
                best_loss = loss
                patience = 0
                save_dir = self.ae_path
                torch.save({
                'state_dict': self.dm_condition.state_dict()
                }, save_dir + '/conditional_edm.pt')
            else:
                patience += 1
                if patience == self.patience:
                    if self.verbose:
                        print('Early stopping')
                    break

    def sample_free(self, condition_model, uncondition_model, data):
        self.ae.eval()
        condition_model.eval()
        uncondition_model.eval()
        condition_net = condition_model.denoise_fn_D
        uncondition_net = uncondition_model.denoise_fn_D
        auc = []
        x = data.x.cuda()
        edge_index = data.edge_index.cuda()
        y = data.y.bool()
        Z_0 = self.ae.encode(x, edge_index)
        ###############  forward process  ####################
        noise = torch.randn_like(Z_0)
        for i in range(0, self.timesteps):
            t = torch.tensor([i] * Z_0.size(0)).long().cuda()
            sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, Z_0.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, Z_0.shape)
            Z_t = sqrt_alphas_cumprod_t * Z_0 + sqrt_one_minus_alphas_cumprod_t * noise

            if self.sample_steps > 0:
                reconstructed = sample_dm_free(condition_net, uncondition_net, Z_t, self.sample_steps, common_feat=self.common_feat, lamda=self.lamda)
            s = to_dense_adj(edge_index)[0].cuda()
            x_, s_ = self.ae.decode(reconstructed, edge_index)
            score = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)

            pyg_auc = eval_roc_auc(y, score.cpu().detach())

            auc.append(pyg_auc)
            print("timestep:{},pyg_AUC: {:.4f}".format(i, pyg_auc))                       

def softmax_with_temperature(input, t=1, axis=-1):
    ex = torch.exp(input/t)
    sum = torch.sum(ex, axis=axis)
    return ex/sum
