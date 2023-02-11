import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch import optim
from CFDA import GraphConvSparse, glorot_init, sparse_dense_mul


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p)
    and sum over the last dimension

    Return: kl between each sample
    """
    # element-wise operation
    # qm, qv, pm, pv = qm.to(device), qv.to(device), pm.to(device), pv.to(device)
    kl = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    # sum over all dimensions except for batch
    kl = kl.sum(-1)
    kl = kl.sum(-1)
    # print("log var1", qv)
    if torch.isnan(kl.any()):
        print("\n\n\n\noveflow\n\n\n\n\n\n")
    return kl


def conditional_sample_gaussian(m, v, device):
    sample = torch.randn(m.size()).to(device)
    z = m.to(device) + (v.to(device) ** 0.5) * sample
    return z


class CausalDAG(nn.Module):
    """
    creates a causal diagram A


    """

    def __init__(self, num_concepts, dim_per_concept, inference=False, bias=False, g_dim=32):

        super(CausalDAG, self).__init__()
        self.num_concepts = num_concepts
        self.dim_per_concept = dim_per_concept

        self.A = nn.Parameter(torch.zeros(num_concepts, num_concepts))
        self.I = nn.Parameter(torch.eye(num_concepts))
        self.I.requires_grad = False
        if bias:
            self.bias = Parameter(torch.Tensor(num_concepts))
        else:
            self.register_parameter('bias', None)

        nets_z = []
        nets_label = []

        for _ in range(num_concepts):
            nets_z.append(
                nn.Sequential(
                    nn.Linear(dim_per_concept, g_dim),
                    nn.ELU(),
                    nn.Linear(g_dim, dim_per_concept)
                )
            )

            nets_label.append(
                nn.Sequential(
                    nn.Linear(1, g_dim),
                    nn.ELU(),
                    nn.Linear(g_dim, 1)
                )
            )
        self.nets_z = nn.ModuleList(nets_z)
        self.nets_label = nn.ModuleList(nets_label)

    def calculate_z(self, epsilon):
        """
        convert epsilon to z using the SCM assumption and causal diagram A

        """

        C = torch.inverse(self.I - self.A.t())

        if epsilon.dim() > 2:  # one concept is represented by multiple dimensions
            z = F.linear(epsilon.permute(0, 2, 1), C, self.bias)
            z = z.permute(0, 2, 1).contiguous()

        else:
            z = F.linear(epsilon, C, self.bias)
        return z

    def calculate_epsilon(self, z):
        """
        convert epsilon to z using the SCM assumption and causal diagram A

        """

        C_inv = self.I - self.A.t()

        if z.dim() > 2:  # one concept is represented by multiple dimensions
            epsilon = F.linear(z.permute(0, 2, 1), C_inv, self.bias)
            epsilon = epsilon.permute(0, 2, 1).contiguous()

        else:
            epsilon = F.linear(z, C, self.bias)
        return epsilon

    def mask(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(dim=-1).cuda()
        res = torch.matmul(self.A.t(), x)
        return res

    def g_z(self, x):
        """
        apply nonlinearity for more stable approximation

        """
        x_flatterned = x.view(-1, self.num_concepts * self.dim_per_concept)
        concepts = torch.split(x_flatterned, self.dim_per_concept, dim=1)
        res = []
        for i, concept in enumerate(concepts):
            t = self.nets_z[i](concept)
            res.append(t)
        x = torch.concat(res, dim=1).reshape([-1, self.num_concepts, self.dim_per_concept])
        return x

    def g_label(self, x):
        """
        apply nonlinearity for more stable approximation

        """
        x_flatterned = x.view(-1, self.num_concepts)
        concepts = torch.split(x_flatterned, 1, dim=1)
        res = []
        for i, concept in enumerate(concepts):
            res.append(self.nets_label[i](concept))
        x = torch.concat(res, dim=1).reshape([-1, self.num_concepts])
        return x

    def forward(self, x):
        return self.g(self.mask(x))


class CFVAE(nn.Module):
    def __init__(self, h_dim, input_dim, adj):
        super(CFVAE, self).__init__()
        self.h_dim = h_dim
        self.s_num = 4
        # A
        self.base_gcn = GraphConvSparse(input_dim, h_dim, adj)
        self.gcn_mean = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.pred_a = nn.Sequential(nn.Linear(h_dim + 1, adj.shape[1]), nn.Sigmoid())
        # X
        self.base_gcn_x = GraphConvSparse(input_dim, h_dim, adj)
        self.gcn_mean_x = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.gcn_logstddev_x = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)

        # reconst_X
        self.reconst_X = nn.Sequential(nn.Linear(h_dim + 1, input_dim))
        # pred_S
        self.pred_s = nn.Sequential(nn.Linear(h_dim + h_dim, self.s_num), nn.Softmax())
        # DAG network
        num_concepts = input_dim
        dim_per_concept = 4
        self.dag = CausalDAG(num_concepts, dim_per_concept)

    def encode_A(self, X):
        mask_X = X
        hidden = self.base_gcn(mask_X)
        mean = self.gcn_mean(hidden)
        logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn_like(mean, requires_grad=True)
        # print(gaussian_noise.size())
        # print(logstd.size())
        if self.training:
            # sampled_z = gaussian_noise * torch.exp(logstd) + mean
            std = torch.exp(0.5 * logstd)
            sampled_z = gaussian_noise * std + mean
        else:
            sampled_z = mean
        return sampled_z

    def encode_X(self, X):
        hidden = self.base_gcn_x(X)
        mean = self.gcn_mean_x(hidden)
        logstd = self.gcn_logstddev_x(hidden)
        gaussian_noise = torch.randn_like(mean, requires_grad=True)
        if self.training:
            # sampled_z = gaussian_noise * torch.exp(logstd) + mean
            std = torch.exp(0.5 * logstd)
            sampled_z = gaussian_noise * std + mean
        else:
            sampled_z = mean
        return sampled_z

    def pred_adj(self, Z, S):
        ZS = torch.cat([Z, S], dim=1)
        A_pred = self.pred_a(ZS)
        # A_pred = F.sigmoid(self.pred_a(ZS, ZS))
        # A_pred = torch.sigmoid(torch.matmul(ZS, ZS.t()))
        return A_pred

    def pred_x(self, Z):
        X_pred = self.reconst_X(Z)
        return X_pred

    def pred_graph(self, Z_a, Z_x, S):
        A_pred = self.pred_adj(Z_a, S)
        X_pred = self.pred_features(Z_x, S)
        return A_pred, X_pred

    def forward(self, X, label):
        # reproduce the causal VAE here
        hidden = self.base_gcn_x(X)
        graph_size = X.size()[0]
        e_m, e_v = self.gcn_mean_x(hidden), self.gcn_logstddev_x(hidden)
        latent_dim = [graph_size, self.num_concepts, self.dim_per_concept]
        e_m, e_v = e_m.reshape(latent_dim), torch.ones(latent_dim).cuda()
        # z = (I - A.T)^(-1) * eps
        z_m, z_v = self.dag.calculate_z(e_m), torch.ones(latent_dim).cuda()

        masked_z_m = self.dag.mask(z_m)
        masked_label = self.dag.mask(label)

        # apply nonlinearity
        masked_z_m = self.dag.g_z(masked_z_m)
        pred_label = self.dag.g_label(masked_label)

        # z for predicting label u
        z = conditional_sample_gaussian(masked_z_m, e_v * self.lambdav, masked_z_m.device)
        rec_x = self.pred_x(z)

        # losses returns by number of samples in the batch

        # reconstruction loss
        rec = F.mse_loss(rec_x, X)

        # KL between eps ~ N(0,1) and Q_phi(eps|x,u)
        p_m, p_v = torch.zeros_like(e_m), torch.ones_like(e_v)
        kl = self.beta * kl_normal(e_m, e_v, p_m, p_v)

        # KL between Q_phi(z|x, u) and P_theta(z|u)
        mean_label = label.mean(dim=0)
        max_label = label.max(dim=0).values
        normalized_label_mean = (label - mean_label) / max_label

        cp_m = einops.repeat(normalized_label_mean, 'b d -> b d repeat', repeat=self.dim_per_concept).cuda()
        cp_v = torch.ones_like(z_v).cuda()
        kl += self.gamma * kl_normal(z_m, z_v, cp_m, cp_v)

        if torch.isnan(kl.mean()):
            print(kl)

        kl = kl.mean()

        # constraints
        lm = kl_normal(z, cp_v, cp_m, cp_v)

        lm = lm.mean()

        lu = F.mse_loss(pred_label.squeeze(dim=-1).cuda(), label.cuda())
        lu = lu.mean()

    def loss_function(self, adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred):
        # loss_reconst
        weighted = True
        if weighted:
            weights_0 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
            weights_1 = 1 - weights_0
            assert (weights_0 > 0 and weights_1 > 0)
            weight = torch.ones_like(A_pred).reshape(-1) * weights_0  # (n x n), weight 0
            idx_1 = adj.to_dense().reshape(-1) == 1
            weight[idx_1] = weights_1

            loss_bce = nn.BCELoss(weight=weight, reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))

        else:
            loss_bce = nn.BCELoss(reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))

        X_ns = X.clone()
        X_ns[:, sen_idx] = 0.  # mute this sensitive dim
        loss_mse = nn.MSELoss(reduction='mean')

        # if self.training and self.type == 'VGAE':
        #     perm = torch.randperm(len(X_ns))
        #     idx = perm[: 1024]
        #     X_ns = X_ns[idx]
        #     X_pred = X_pred[idx]
        loss_reconst_x = loss_mse(X_pred, X_ns)

        loss_ce = nn.CrossEntropyLoss()
        loss_s = loss_ce(S_agg_pred, S_agg_cat.view(-1))  # S_agg_pred: n x K, S_agg: n
        loss_result = {'loss_reconst_a': loss_reconst_a, 'loss_reconst_x': loss_reconst_x, 'loss_s': loss_s}
        return loss_result

    def train_model(self, X, adj, sen_idx, dataset, model_path='', lr=0.0001, weight_decay=1e-5):
        rate_1 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
        print('adj=1: ', rate_1)

        par_s = list(self.pred_s.parameters())
        par_other = list(self.base_gcn.parameters()) + list(self.gcn_mean.parameters()) + list(
            self.gcn_logstddev.parameters()) + list(self.pred_a.parameters()) + \
                    list(self.base_gcn_x.parameters()) + list(self.gcn_mean_x.parameters()) + list(
            self.gcn_logstddev_x.parameters()) + list(self.reconst_X.parameters())
        optimizer_1 = optim.Adam([{'params': par_s, 'lr': lr}], weight_decay=weight_decay)  #
        optimizer_2 = optim.Adam([{'params': par_other, 'lr': lr}], weight_decay=weight_decay)  #

        self.train()
        n = X.shape[0]

        S = X[:, sen_idx].view(-1, 1)  # n x 1
        S_agg = torch.mm(adj, S) / n  # n x 1
        S_agg_max = S_agg.max()
        S_agg_min = S_agg.min()
        S_agg_cat = torch.floor(S_agg / ((S_agg_max + 0.000001 - S_agg_min) / self.s_num)).long()  # n x 1

        print("start training counterfactual augmentation module!")
        for epoch in range(2000):
            for i in range(3):
                optimizer_1.zero_grad()

                A_pred, X_pred, S_agg_pred = self.forward(X, sen_idx)
                loss_result = self.loss_function(adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred)

                # backward propagation
                loss_s = loss_result['loss_s']
                loss_s.backward()
                optimizer_1.step()

            for i in range(5):
                optimizer_2.zero_grad()
                A_pred, X_pred, S_agg_pred = self.forward(X, sen_idx)
                loss_result = self.loss_function(adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred)

                # backward propagation
                loss_s = loss_result['loss_s']
                loss_reconst_x = loss_result['loss_reconst_x']
                loss_reconst_a = loss_result['loss_reconst_a']
                # loss_reconst_a.backward()
                (-loss_s + loss_reconst_a + loss_reconst_x).backward()
                optimizer_2.step()

            if epoch % 100 == 0:
                self.eval()
                eval_result = self.test(adj, X, sen_idx, S_agg_cat)
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_reconst_a: {:.4f}'.format(loss_reconst_a.item()),
                      'loss_reconst_x: {:.4f}'.format(loss_reconst_x.item()),
                      'loss_s: {:.4f}'.format(loss_s.item()),
                      'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                      'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                      'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
                      )
                # save model
                save_model = True
                if save_model and epoch > 0:
                    save_model_path = model_path + f'weights_CFDA_{dataset}_{self.type}' + '.pt'
                    torch.save(self.state_dict(), save_model_path)
                    print('saved model weight in: ', save_model_path)
                self.train()
        return

    def test(self, adj, X, sen_idx, S_agg_cat):
        self.eval()
        A_pred, X_pred, S_agg_pred = self.forward(X, sen_idx)
        loss_result = self.loss_function(adj, X, sen_idx, S_agg_cat, A_pred, X_pred, S_agg_pred)
        eval_result = loss_result

        A_pred_binary = (A_pred > 0.5).float()  # binary
        adj_size = A_pred_binary.shape[0] * A_pred_binary.shape[1]

        sum_1 = torch.sparse.sum(adj)
        correct_num_1 = torch.sparse.sum(sparse_dense_mul(adj, A_pred_binary))  # 1
        correct_num_0 = (adj_size - (A_pred_binary + adj).sum() + correct_num_1)
        acc_a_pred = (correct_num_1 + correct_num_0) / adj_size
        acc_a_pred_0 = correct_num_0 / (adj_size - sum_1)
        acc_a_pred_1 = correct_num_1 / sum_1

        eval_result['acc_a_pred'] = acc_a_pred
        eval_result['acc_a_pred_0'] = acc_a_pred_0
        eval_result['acc_a_pred_1'] = acc_a_pred_1

        eval_result = loss_result
        eval_result['acc_a_pred'] = acc_a_pred
        eval_result['acc_a_pred_0'] = acc_a_pred_0
        eval_result['acc_a_pred_1'] = acc_a_pred_1
        return eval_result
