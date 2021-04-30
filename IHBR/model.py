

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import att_layer

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values),
                                     torch.Size(graph.shape))
    return graph

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out




class IHBR(nn.Module):

    def __init__(self, args,item_num, user_num, bundle_num, input_graph, device, act = nn.ReLU()):
        super(IHBR, self).__init__()
        self.act = act
        self.dropout = args.dropout
        self.layers = args.layers
        self.emb = args.embedding
        self.dropout = nn.Dropout(args.dropout, True)
        self.num_items = item_num
        self.num_bundles = bundle_num
        self.num_users = user_num
        self.device = device


        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.emb))
        nn.init.xavier_normal_(self.users_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.emb))
        nn.init.xavier_normal_(self.items_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.emb))
        nn.init.xavier_normal_(self.bundles_feature)


        self.predictlayer = PredictLayer(3 * 3 * self.emb, 0)
        self.matatt = att_layer.MHAtt(8, self.emb, self.emb, self.emb)
        self.dnns_ub = nn.ModuleList([nn.Linear(
            self.emb * (l + 1), self.emb) for l in range(self.layers)])
        self.dnns_ui = nn.ModuleList([nn.Linear(
            self.emb * (l + 1), self.emb) for l in range(self.layers)])
        self.att_i = nn.ModuleList([att_layer.Att(self.emb * (l + 1)) for l in range(self.layers)])
        self.att_b = nn.ModuleList([att_layer.Att(self.emb * (l + 1)) for l in range(self.layers)])

        ub_graph, ui_graph, bi_graph = input_graph
        i_u_i = ui_graph.T @ ui_graph
        i_b_i = bi_graph.T @ bi_graph
        A_i = sp.diags(1 / ((i_u_i.sum(axis=1) + 1e-8).A.ravel())) @ i_u_i
        B_i = sp.diags(1 / ((i_b_i.sum(axis=1) + 1e-8).A.ravel())) @ i_b_i

        b_u_i = ub_graph.T @ ui_graph
        C_b = sp.diags(1 / ((b_u_i.sum(axis=1) + 1e-8).A.ravel())) @ b_u_i
        self.C_b = to_tensor(C_b).to(self.device)

        ub_norm = sp.diags(1 / ((ub_graph.sum(axis=1) + 1e-8).A.ravel())) @ ub_graph
        bu_norm = sp.diags(1 / ((ub_graph.sum(axis=0) + 1e-8).A.ravel())) @ ub_graph.T
        ui_norm = sp.diags(1 / ((ui_graph.sum(axis=1) + 1e-8).A.ravel())) @ ui_graph
        iu_norm = sp.diags(1 / ((ui_graph.sum(axis=0) + 1e-8).A.ravel())) @ ui_graph.T
        bi_norm = sp.diags(1 / ((bi_graph.sum(axis=1) + 1e-8).A.ravel())) @ bi_graph
        ib_norm = sp.diags(1 / ((bi_graph.sum(axis=0) + 1e-8).A.ravel())) @ bi_graph.T

        x = bi_norm @ A_i
        x_norm = sp.diags(1 / ((x.sum(axis=1) + 1e-8).A.ravel())) @ x
        self.A = to_tensor(x_norm).to(self.device)
        self.A_i = to_tensor(A_i).to(self.device)

        y = bi_norm @ B_i
        y_norm = sp.diags(1 / ((y.sum(axis=1) + 1e-8).A.ravel())) @ y
        self.B = to_tensor(y_norm).to(self.device)
        self.B_i = to_tensor(B_i).to(self.device)

        tmpub_graph = sp.bmat([[sp.identity(ub_graph.shape[0]), ub_norm],
                               [bu_norm, sp.identity(ub_graph.shape[1])]])
        self.ub_graph = to_tensor(laplace_transform(tmpub_graph)).to(self.device)

        tmpui_graph = sp.bmat([[sp.identity(ui_graph.shape[0]), ui_norm],
                               [iu_norm, sp.identity(ui_graph.shape[1])]])
        self.ui_graph = to_tensor(laplace_transform(tmpui_graph)).to(self.device)

        self.bi_avg = to_tensor(bi_norm).to(self.device)
        self.ui_avg = to_tensor(ui_norm).to(self.device)
        self.ub_avg = to_tensor(ub_norm).to(self.device)
        self.ib_avg = to_tensor(ib_norm).to(self.device)

    def Intention(self):
        items_iui = self.act(torch.matmul(self.A_i, self.items_feature))
        items_ibi = self.act(torch.matmul(self.B_i, self.items_feature))
        items = items_iui + items_ibi + self.items_feature
        items, a_i = self.matatt(items, items, items)

        bundles_on_bi = self.act(torch.matmul(self.bi_avg, items))
        bundles = bundles_on_bi + self.bundles_feature

        users_on_ui = self.act(torch.matmul(self.ui_avg, items))
        users_on_bi = self.act(torch.matmul(self.ub_avg, bundles))
        users = users_on_ui + users_on_bi + self.users_feature

        return users, bundles, items


    def NPPT(self, ui_graph,ub_graph, u_feature, b_feature, i_feature,dnns_ui,dnns_ub):
        features_ub = torch.cat((u_feature, b_feature), 0)
        all_features_ub = [features_ub]
        features_ui = torch.cat((u_feature, i_feature), 0)
        all_features_ui = [features_ui]

        for i in range(self.layers):
            cur_ub = features_ub
            cur_ui = features_ui
            features_ub = self.act(dnns_ub[i](torch.matmul(ub_graph, cur_ub)))
            features_ui = self.act(dnns_ui[i](torch.matmul(ui_graph, cur_ui)))

            u1, i1 = torch.split(
                cur_ui, (u_feature.shape[0], i_feature.shape[0]), 0)
            b1 = F.normalize(self.act(torch.matmul(self.bi_avg, i1)))
            x = torch.cat((u1,b1),0)
            ub = torch.stack([cur_ub,x], dim=1)
            ub = self.att_b[i](ub)

            features_ub = torch.cat([features_ub, ub], 1)
            features_ub = self.dropout(features_ub)

            u2, b2 = torch.split(
                cur_ub, (u_feature.shape[0], b_feature.shape[0]), 0)
            i2 = F.normalize(self.act(torch.matmul(self.ib_avg, b2)))
            y = torch.cat((u2,i2),0)
            ui = torch.stack([cur_ui,y],dim=1)
            ui = self.att_i[i](ui)

            features_ui= torch.cat([features_ui,ui], 1)
            features_ui = self.dropout(features_ui)

            all_features_ub.append(F.normalize(features_ub))
            all_features_ui.append(F.normalize(features_ui))



        all_features_ub = torch.cat(all_features_ub, 1)
        all_features_ui = torch.cat(all_features_ui, 1)

        A_feature, B_feature= torch.split(
            all_features_ub, (u_feature.shape[0], b_feature.shape[0]), 0)
        C_feature, D_feature= torch.split(
            all_features_ui, (u_feature.shape[0], i_feature.shape[0]), 0)

        return A_feature,  B_feature, C_feature, D_feature


    def model(self):
        users, bundles, items = self.Intention()
        ub_users_feature, ub_bundles_feature, ui_users_feature, ui_items_feature = self.NPPT(
            self.ui_graph,self.ub_graph, users, bundles, items, self.dnns_ui,self.dnns_ub)

        return ub_users_feature, ub_bundles_feature

    def predict(self, users_feature, bundles_feature):
        element_emb = torch.mul(bundles_feature,users_feature)
        new_emb = torch.cat((element_emb,bundles_feature,users_feature),dim=1)
        y = self.predictlayer(new_emb)

        return y

    def forward(self, users, bundles):
        users_feature, bundles_feature = self.model()
        users_embedding = users_feature[users]
        bundles_embedding = bundles_feature[bundles]
        pred = self.predict(users_embedding, bundles_embedding)
        return pred




