
import data_prep
import dataset
import eval_metrics
import torch.utils.data as data
from model import IHBR
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id",
    type=str,
    default="0",
    help="gpu")
parser.add_argument("--dropout",
    type=float,
    default=0.3,
    help="dropout rate")
parser.add_argument("--embedding",
    type=int,
    default=64,
    help="embedding size")
parser.add_argument("--layers",
    type=int,
    default=1,
    help="number of layers")
parser.add_argument("--batch_size",
    type=int,
    default=4096,
    help="batch size")
parser.add_argument("--epochs",
    type=int,
    default=50,
    help="epoches")
parser.add_argument("--top_k",
    type=int,
    default=5,
    help="top_k")
parser.add_argument("--train_neg_num",
    type=int,
    default=2,
    help="training negative items")
parser.add_argument("--test_neg_num",
    type=int,
    default=99,
    help="testing negative items")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
cudnn.benchmark = True
device = torch.device('cuda')

def train(model, epochs, loader, optim, device, loss_func, test_loader):
    for epoch in range(epochs):
        model.train()
        loader.dataset.neg_sample()
        for user, bundle, label in loader:
            label = label.float().to(device)
            optim.zero_grad()
            modelout = model(user.to(device), bundle.to(device))
            modelout = modelout.view(-1)
            loss = loss_func(modelout, label)
            loss.backward()
            optim.step()

        model.eval()
        HR, NDCG, MRR = eval_metrics.get_metrics(model, test_loader, args.top_k,device)

        print("HR: {:.4f}\tMRR: {:.4f}\tNDCG: {:.4f}\t".format(np.mean(HR), np.mean(MRR), np.mean(NDCG)))

    return loss

def get_graph(user_bundle_data,user_item_data,bundle_item_data,item_num,user_num,bundle_num):
    ub_indice = np.array(user_bundle_data, dtype=np.int32)
    ub_values = np.ones(len(user_bundle_data), dtype=np.float32)
    ground_truth_u_b = sp.coo_matrix(
        (ub_values, (ub_indice[:, 0], ub_indice[:, 1])), shape=(user_num, bundle_num)).tocsr()

    ui_indice = np.array(user_item_data, dtype=np.int32)
    ui_values = np.ones(len(user_item_data), dtype=np.float32)
    ground_truth_u_i = sp.coo_matrix(
        (ui_values, (ui_indice[:, 0], ui_indice[:, 1])), shape=(user_num, item_num)).tocsr()

    bi_indice = np.array(bundle_item_data, dtype=np.int32)
    bi_values = np.ones(len(bundle_item_data), dtype=np.float32)
    ground_truth_b_i = sp.coo_matrix(
        (bi_values, (bi_indice[:, 0], bi_indice[:, 1])), shape=(bundle_num, item_num)).tocsr()

    return ground_truth_u_b, ground_truth_u_i, ground_truth_b_i

def main():
    train_data,test_data,user_bundle_data,user_item_data,bundle_item_data,\
    item_num,user_num,bundle_num,user_bundle_mat = dataset.load_dataset()
    train_dataset = data_prep.CreateData(
        train_data, bundle_num, user_bundle_mat, args.train_neg_num, True)
    test_dataset = data_prep.CreateData(
        test_data, bundle_num, user_bundle_mat, 0, False)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=99 + 1, shuffle=False, num_workers=0)

    ub_graph, ui_graph, bi_graph = get_graph(train_data,user_item_data,bundle_item_data,item_num,user_num,bundle_num)
    graph = [ub_graph, ui_graph, bi_graph]

    #print(args)

    model = IHBR(args, item_num, user_num, bundle_num, graph, device).to(device)
    op = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-7)
    loss_func = nn.BCEWithLogitsLoss()
    loss = train(model, args.epochs, train_loader, op, device,loss_func,test_loader)


if __name__ == '__main__':
    main()