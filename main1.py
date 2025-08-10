import argparse
from data_loader import *
from torch.optim.lr_scheduler import CyclicLR
from model import *
from sklearn import metrics
from utils1 import *
from utils import *
from torch_scatter.scatter import scatter_add
import sys

import json
import os
import time

import logging


sys.path.append(".")

def parse_args():
    parser = argparse.ArgumentParser(description="Run DRAGON.")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='Fdataset', help='Choose a dataset:[Fdataset, Cdataset, LRSSL]')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024*5, help='Batch size.')
    parser.add_argument('--weight_decay', type=float, default=0.0001) # 0.0001
    parser.add_argument('--disease_TopK', type=int, default=4)
    parser.add_argument('--drug_TopK', type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=8)
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument("--model", type=str, default="Panacea", help="backbone")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--trend_coeff', type=float, default=1, help='coefficient of attention')
    parser.add_argument("--n_hops", type=int, default=2)
    parser.add_argument("--aggr", type=str, default='mean')
    parser.add_argument("--layer_sizes", nargs='?', default=[64, 64, 64])
    parser.add_argument('--bt_coeff', type=float, default=0.01, help='learning rate')
    parser.add_argument('--all_bt_coeff', type=float, default=0.2, help='learning rate')

    parser.add_argument('--emb_coeff', type=float, default=0, help='coefficient for emb_weight and emb_weight2')

    parser.add_argument('--tm_net_hidden_dimension', type=int, default=256, help='tm_net hidden dimension')
    # augmentation
    parser.add_argument('--edge_drop', type=float, default=0.12, help='Edge dropout ratio')
    parser.add_argument('--feat_mask', type=float, default=0.12, help='Feature mask ratio')

    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# written log
def setup_logging(gpu_id):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = './test_result'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    timestamp_dir = os.path.join(result_dir, timestamp)
    if not os.path.exists(timestamp_dir):
        os.makedirs(timestamp_dir, exist_ok=True)
    
    log_file = os.path.join(timestamp_dir, f"gpu_{gpu_id}_log.txt")

    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    return logging

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

if __name__ == "__main__":
    # avg_auroc, avg_aupr = [], []
    gpu_id = args.gpu_id
    logger = setup_logging(gpu_id)
    
    logger.info(f"Starting {gpu_id + 1} th 10-fold validation...")


    setup_seed(args.gpu_id)
    disease_adj, drug_adj, original_interactions, all_train_mask, all_test_mask, pos_weight = data_preparation(args)
    all_scores, all_labels = [], []
    # print(f'+++++++++++++++This is {args.gpu_id + 1}-th 10 fold validation.+++++++++++++++')
    for fold_num in range(len(all_train_mask)):
        logger.info(f"---------------This is {fold_num + 1}-th fold validation.---------------")
        # print(f'---------------This is {fold_num + 1}-th fold validation.---------------')

        # dataset splitting
        train_manager, test_manager = data_split(args, all_train_mask[fold_num], all_test_mask[fold_num],
                                                    original_interactions)
        train_adj = train_manager.train_adj

        """build model"""
        model = Panacea(args, (train_manager, train_adj, disease_adj, drug_adj, pos_weight)).to(args.device)

        """define optimizer"""
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        lr_scheduler = CyclicLR(optimizer, base_lr=0.1 * args.lr, max_lr=args.lr, step_size_up=20,
                                mode="exp_range", gamma=0.995, cycle_momentum=False)

        adj_sp_norm = model.adj_mat
        adj_sp_norm2 = model.adj_mat2

        edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()
        edge_index2, edge_weight2 = adj_sp_norm2._indices(), adj_sp_norm2._values()

        model.adj_sp_norm = adj_sp_norm.to(args.device)
        model.edge_index = edge_index.to(args.device)
        model.edge_weight = edge_weight.to(args.device)

        model.adj_sp_norm2 = adj_sp_norm2.to(args.device)
        model.edge_index2 = edge_index2.to(args.device)
        model.edge_weight2 = edge_weight2.to(args.device)

        row, col = edge_index
        row2, col2 = edge_index2

        # Multimodal
        # Compute adj_sp_norm's embedding weight
        src_emb = model.emb1[row] # [num_edges, 8]
        tgt_emb = model.emb1[col] # [num_edges, 8]

        # Compute cosine similarity
        emb_weight = F.cosine_similarity(src_emb, tgt_emb, dim=1) # [E]

        # Compute adj_sp_norm2's embedding weight
        src_emb2 = model.emb2[row2] # [num_edges2, 8]
        tgt_emb2 = model.emb2[col2] # [num_edges2, 8]

        # Compute cosine similarity
        emb_weight2 = F.cosine_similarity(src_emb2, tgt_emb2, dim=1) # [E2]

        
        cal_adaptive_weight = compute_adaptive_edge_weight
        cal_adaptive_weight2 = compute_adaptive_edge_weight2
        # cal_adaptive_weight = co_ratio_deg_disease_sc
        # cal_adaptive_weight2 = co_ratio_deg_disease_sc2

        adaptive_edge_weight = cal_adaptive_weight(adj_sp_norm, edge_index, args)
        adaptive_edge_weight2 = cal_adaptive_weight2(adj_sp_norm2, edge_index2, args)

        # print("===================")
        # print(f"trend1: {adaptive_edge_weight}")
        # print(f"trend2: {adaptive_edge_weight2}")
        # print("===================")

        adaptive_edge_weight = adaptive_edge_weight.to(args.device)
        col = col.to(args.device)

        norm_now = scatter_add(
            adaptive_edge_weight, col, dim=0, dim_size=args.n_diseases + args.n_drugs)[col]

        adaptive_edge_weight2 = adaptive_edge_weight2.to(args.device)
        col2 = col2.to(args.device)

        norm_now2 = scatter_add(
            adaptive_edge_weight2, col2, dim=0, dim_size=args.n_diseases + args.n_drugs)[col2]

        adaptive_edge_weight = args.trend_coeff * adaptive_edge_weight / norm_now + edge_weight + args.emb_coeff * emb_weight
        adaptive_edge_weight2 = args.trend_coeff * adaptive_edge_weight2 / norm_now2 + edge_weight2 + args.emb_coeff * emb_weight2

        model.adap_weight = (adaptive_edge_weight).to(args.device)
        model.adap_weight2 = (adaptive_edge_weight2).to(args.device)

        for epoch in range(args.epochs):
            model.train()
            loss_list = []
            for batch in train_manager.iter_batch(shuffle=True):
                loss, _ = model.forward(batch)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Parameter {name} update gradient: {param.grad.norm()}")

                optimizer.step()
                lr_scheduler.step()
            model.eval()
            scores, labels = [], []
            for batch in test_manager.iter_batch():
                score, label = model.generate(batch)
                scores.append(score.cpu().detach().numpy())
                labels.append(label)
            loss_sum = np.sum(loss_list)
            scores = np.concatenate(scores)
            labels = np.concatenate(labels)
            aupr = metrics.average_precision_score(y_true=labels, y_score=scores)
            auroc = metrics.roc_auc_score(y_true=labels, y_score=scores)
            # print(f'Epoch: {epoch + 1}, auroc: {auroc}, aupr: {aupr}')
            logger.info(f"Epoch: {epoch + 1}, auroc: {auroc}, aupr: {aupr}")
            if (epoch + 1) == args.epochs:
                all_scores.append(scores)
                all_labels.append(labels)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    aupr = metrics.average_precision_score(y_true=all_labels, y_score=all_scores)
    auroc = metrics.roc_auc_score(y_true=all_labels, y_score=all_scores)
    # avg_auroc.append(auroc)
    # avg_aupr.append(aupr)
    logger.info(f"------------------------------------------------------------------------")
    logger.info(f"{args.gpu_id + 1}-th 10 cv auroc：{auroc:.5f}")
    logger.info(f"{args.gpu_id + 1}-th 10 cv aupr：{aupr:.5f}")