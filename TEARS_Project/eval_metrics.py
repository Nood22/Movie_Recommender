import numpy as np

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx_topk_part = np.argpartition(-X_pred, k, axis=1)[:, :k]
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    X_true_binary = (heldout_batch > 0).astype(np.float32)
    X_pred_binary = np.zeros_like(X_true_binary, dtype=np.float32)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx_topk] = 1
    tmp = np.logical_and(X_true_binary, X_pred_binary).sum(axis=1).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    recall[np.isnan(recall)] = 0.0
    return np.mean(recall)

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx_topk_part = np.argpartition(-X_pred, k, axis=1)[:, :k]
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    X_true_binary = (heldout_batch > 0).astype(np.float32)
    DCG = (X_true_binary[np.arange(batch_users)[:, np.newaxis], idx_topk] /
           np.log2(np.arange(2, k + 2))).sum(axis=1)
    IDCG = np.array([(1.0 / np.log2(np.arange(2, min(n + 1, k) + 2))).sum()
                     for n in X_true_binary.sum(axis=1)])
    IDCG[IDCG == 0.0] = 1.0
    ndcg = DCG / IDCG
    return np.mean(ndcg)

def MRR_at_k(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx_topk_part = np.argpartition(-X_pred, k, axis=1)[:, :k]
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    X_true_binary = (heldout_batch > 0).astype(np.float32)

    mrrs = []
    for i in range(batch_users):
        for rank, item in enumerate(idx_topk[i]):
            if X_true_binary[i, item]:
                mrrs.append(1.0 / (rank + 1))
                break
        else:
            mrrs.append(0.0)
    return np.mean(mrrs)
