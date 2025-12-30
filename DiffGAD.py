import os
import tqdm
import math
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import BaseTransform
from sklearn.metrics import auc, precision_recall_curve
from pygod.utils import load_data
from pygod.metric.metric import eval_roc_auc, eval_average_precision, eval_recall_at_k

from auto_encoder import GraphAE
from utils import extract, softmax_with_temperature, get_noises
from diffusion_model import MLPDiffusion, Model, sample_dm_free

# calculations for noises
sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_noises(timesteps=500)

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
                 weight=0.0,
                 sample_steps=50,
                 ae_dropout=0.3,
                 ae_lr=0.05, 
                 ae_alpha=0.8,
                 proto_alpha = None,
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
        self.weight = weight
        
        self.proto = None
        self.dm = None
        self.proto_alpha = proto_alpha

        self.ae = None
        self.ae_dropout = ae_dropout
        self.ae_lr = ae_lr
        self.ae_alpha = ae_alpha
        self.ae_ckpt = None
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.timesteps = 500

    def forward(self, dset):
        self.dataset = dset
        data = load_data(self.dataset)

        if self.hid_dim is None:
            self.hid_dim = 2 ** int(math.log2(data.x.size(1)) - 1)
        if self.diff_dim is None:
            self.diff_dim = 2 * self.hid_dim
        
        self.ae = GraphAE(in_dim = data.num_node_features, 
                           hid_dim=self.hid_dim,
                           dropout=self.ae_dropout).cuda()
        self.save_dir = os.getcwd() + '/models/'+ self.dataset + '/full_batch/'
        self.ae_path = self.save_dir + "/" + str(self.ae_dropout) + "_" + str(self.ae_lr) + "_" + str(self.ae_alpha) + "_" + str(self.hid_dim)
        if not os.path.exists(self.ae_path):
            os.makedirs(self.ae_path)
        ####################### train autoencoder ######################
        if self.ae_ckpt is None:
            self.ae_ckpt = self.train_ae(data)
        ######################## load autoencoder ######################
        print("loading checkpoint from %04d" % self.ae_ckpt)
        ae_dict = torch.load(self.ae_path + '/' + str(self.ae_ckpt) + '.pt')
        self.ae.load_state_dict(ae_dict['state_dict'])
        ###################### train diffusion model #####################
        num_trial = 20
        dm_auc, dm_ap, dm_rec, dm_auprc = [], [], [], []
        for _ in tqdm.tqdm(range(num_trial)):
            #################################
            # original diffusion model
            denoise_fn = MLPDiffusion(self.hid_dim, self.diff_dim).cuda()
            self.dm = Model(denoise_fn=denoise_fn,
                            hid_dim=self.hid_dim).cuda()
            self.proto = self.train_dm(data)
            dm_dict = torch.load(self.ae_path + '/dm.pt')
            self.dm.load_state_dict(dm_dict['state_dict'])
            self.proto = dm_dict['prototype']
            #################################
            # conditional diffusion model
            denoise_proto = MLPDiffusion(self.hid_dim, self.diff_dim).cuda()
            self.dm_proto = Model(denoise_fn=denoise_proto,
                            hid_dim=self.hid_dim).cuda()
            self.train_dm_proto(data)
            dm_free_dict = torch.load(self.ae_path + '/proto_dm.pt')
            self.dm_proto.load_state_dict(dm_free_dict['state_dict'])
            #################################
            # evaluation
            print("weight=", "{:.2f}".format(self.weight))
            auc_this_trial, ap_this_trial, rec_this_trial, auprc_this_trial = self.sample(self.dm_proto, self.dm, data)
            dm_auc.append(auc_this_trial)
            dm_ap.append(ap_this_trial)
            dm_rec.append(rec_this_trial)
            dm_auprc.append(auprc_this_trial)

        dm_auc = torch.tensor(dm_auc)
        dm_ap  = torch.tensor(dm_ap)
        dm_rec = torch.tensor(dm_rec)
        dm_auprc = torch.tensor(dm_auprc)

        print("Final AUC: {:.4f}±{:.4f} ({:.4f})\t"
              "Final AP: {:.4f}±{:.4f} ({:.4f})\t"
              "Final Recall: {:.4f}±{:.4f} ({:.4f})\t"
              "Final AUPRC: {:.4f}±{:.4f} ({:.4f})".format(torch.mean(dm_auc),
                                                torch.std(dm_auc),
                                                torch.max(dm_auc),
                                                torch.mean(dm_ap),
                                                torch.std(dm_ap),
                                                torch.max(dm_ap),
                                                torch.mean(dm_rec),
                                                torch.std(dm_rec),
                                                torch.max(dm_rec),
                                                torch.mean(dm_auprc),
                                                torch.std(dm_auprc),
                                                torch.max(dm_auprc)))

    def train_ae(self, data):
        num_trial = 20
        if self.verbose:
            print('Training autoencoder ...')
        lr = self.ae_lr
        optimizer = torch.optim.Adam(self.ae.parameters(), lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)   
        auc, ap, rec = [], [], []
        
        for _ in tqdm.tqdm(range(num_trial)):
            for epoch in range(1, self.ae_epochs+1):
                self.ae.train()
                optimizer.zero_grad()

                x = data.x.cuda()
                x = x.to(torch.float32)
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

                pyg_auc = eval_roc_auc(y, score.cpu().detach())
                pyg_ap = eval_average_precision(y, score.cpu().detach())
                pyg_rec = eval_recall_at_k(y, score.cpu().detach(), sum(y))

                if epoch%100 == 0:
                    print("Epoch:", '%04d' % (epoch), "loss=", "{:.5f}".format(loss))
                    print("pyg_AUC: {:.4f}, pyg_AP: {:.4f}, pyg_Recall: {:.4f}".format(pyg_auc, pyg_ap, pyg_rec))

            auc.append(pyg_auc)        
            ap.append(pyg_ap)
            rec.append(pyg_rec)
            save_path = self.ae_path
            torch.save({
                'state_dict': self.ae.state_dict(),
            }, save_path + '/'+str(_)+'.pt')

        auc_id = np.argmax(auc)
        auc = torch.tensor(auc)
        ap  = torch.tensor(ap)
        rec = torch.tensor(rec)

        print("AUC: {:.4f}±{:.4f} ({:.4f})\t"
              "AP: {:.4f}±{:.4f} ({:.4f})\t"
              "Recall: {:.4f}±{:.4f} ({:.4f})".format(torch.mean(auc),
                                                torch.std(auc),
                                                torch.max(auc),
                                                torch.mean(ap),
                                                torch.std(ap),
                                                torch.max(ap),
                                                torch.mean(rec),
                                                torch.std(rec),
                                                torch.max(rec)))
        return auc_id

    def train_dm(self, data):
        if self.verbose:
            print('Training diffusion model ...')      
        optimizer = torch.optim.Adam(self.dm.parameters(), lr=self.lr,
                                     weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 
        self.dm.train()
        best_loss = float('inf') 

        patience = 0
        proto = None
        for epoch in range(self.diff_epochs):
            x = data.x.cuda()
            x = x.to(torch.float32)
            edge_index = data.edge_index.cuda()
            inputs = self.ae.encode(x, edge_index) 

            if epoch == 0:
                proto = torch.mean(inputs, dim=0)
            else:
                s_v = self.cos(proto, reconstructed)
                weight = softmax_with_temperature(s_v,t=5).reshape(1,-1)
                proto = torch.mm(weight, reconstructed).detach() 

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
                'prototype': proto
                }, save_dir + '/dm.pt')
            else:
                patience += 1
                if patience == self.patience:
                    if self.verbose:
                        print('Early stopping')
                    break

        return proto
    
    def train_dm_proto(self, data):
        if self.verbose:
            print('Training conditional diffusion model ...')      
        optimizer = torch.optim.Adam(self.dm_proto.parameters(), lr=self.lr,
                                     weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 
        self.dm_proto.train()
        best_loss = float('inf') 

        patience = 0
        for epoch in range(self.diff_epochs):
            x = data.x.cuda()
            x = x.to(torch.float32)
            edge_index = data.edge_index.cuda()

            inputs = self.ae.encode(x, edge_index)
            loss, score_train, reconstructed = self.dm_proto(inputs, proto=self.proto, proto_alpha = self.proto_alpha)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dm_proto.parameters(), 1.0)
            optimizer.step()
            if epoch%10 == 0:
                print("Epoch:", '%04d' % (epoch), "loss=", "{:.5f}".format(loss))
            scheduler.step()

            if loss < best_loss:
                best_loss = loss
                patience = 0
                save_dir = self.ae_path
                torch.save({
                'state_dict': self.dm_proto.state_dict()
                }, save_dir + '/proto_dm.pt')
            else:
                patience += 1
                if patience == self.patience:
                    if self.verbose:
                        print('Early stopping')
                    break


    def sample(self, proto_model, free_model, data):
        self.ae.eval()
        proto_model.eval()
        free_model.eval()
        proto_net = proto_model.denoise_fn_D
        free_net = free_model.denoise_fn_D

        x = data.x.cuda()
        x = x.to(torch.float32)
        edge_index = data.edge_index.cuda()
        y = data.y.bool()
        Z_0 = self.ae.encode(x, edge_index)
        ###############  forward process  ####################
        noise = torch.randn_like(Z_0)
        auc_pygod, ap, rec, auprc = [], [], [], []

        for i in range(0, self.timesteps):
            t = torch.tensor([i] * Z_0.size(0)).long().cuda()
            sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, Z_0.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, Z_0.shape)
            Z_t = sqrt_alphas_cumprod_t * Z_0 + sqrt_one_minus_alphas_cumprod_t * noise

            if self.sample_steps > 0:
                reconstructed = sample_dm_free(proto_net, free_net, Z_t, self.sample_steps, proto=self.proto, proto_alpha = self.proto_alpha, weight=self.weight)
            s = to_dense_adj(edge_index)[0].cuda()
            x_, s_ = self.ae.decode(reconstructed, edge_index)
            score = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)

            pyg_auc = eval_roc_auc(y, score.cpu().detach())
            pyg_ap = eval_average_precision(y, score.cpu().detach())
            pyg_rec = eval_recall_at_k(y, score.cpu().detach(), sum(y))
            auc_pygod.append(pyg_auc)
            ap.append(pyg_ap)
            rec.append(pyg_rec)

            p, r, _ = precision_recall_curve(y.numpy(), score.detach().cpu().numpy())
            auprc.append(auc(r, p))

            print("timestep:{},pyg_AUC: {:.4f}, pyg_AP: {:.4f}, pyg_Recall: {:.4f}, AUPRC: {:.4f}".format(i, pyg_auc, pyg_ap, pyg_rec, auc(r, p)))                      

        this_auc = np.max(auc_pygod)
        this_ap = np.max(ap)
        this_rec = np.max(rec)
        this_auprc = np.max(auprc)
        return this_auc, this_ap, this_rec, this_auprc

