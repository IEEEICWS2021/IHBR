import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0

def map(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return 1 / (index+1)
	return 0

def mrr(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return 1 / (index + 1)
	return 0

def get_metrics(model, test_loader, top_k,device):
	HR, NDCG, MRR = [], [], []

	for user, bundle, label in test_loader:
		pred = model(user.to(device), bundle.to(device))
		pred = pred.view(-1)

		_, idx = torch.topk(pred, top_k)
		recommends = torch.take(bundle.to(device), idx).cpu().numpy().tolist()

		true = bundle[0].item()
		HR.append(hit(true, recommends))

		NDCG.append(ndcg(true, recommends))
		MRR.append(mrr(true, recommends))

	return np.mean(HR), np.mean(NDCG), np.mean(MRR)
