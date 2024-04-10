import torch
import numpy as np
from . import constant as C

class TopKAccuracy:
    def __init__(self):
        self.metrics = {}

    def add(self, k, accuracy):
        assert isinstance(k, int) and k > 0, f'Invalid k: {k}. '
        assert 0 <= accuracy <= 100 or np.isnan(accuracy), f'Invalid accuracy: {accuracy}. '
        self.metrics[k] = accuracy

    def __str__(self):
        return ''.join(f'Top-{k:2d}: {accuracy:.3f}% ' for k, accuracy in self.metrics.items())

def topk_accuracy(similarity, y, topk=C.EVALUATE_TOP_K):
    maxk = max(topk)

    pred = similarity.argsort(axis=1, descending=True)[:, :maxk]
    correct = pred == y[:, None]

    metric = TopKAccuracy()

    for k in topk:
        correct_k = correct[:, :k]
        correct_k = correct_k.sum(axis=1, dtype=torch.float32)
        accuracy = correct_k.mean() * 100
        accuracy = accuracy.item()
        metric.add(k, accuracy)
    return metric
