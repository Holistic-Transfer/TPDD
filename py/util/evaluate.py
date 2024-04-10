import copy
import torch
import torch.nn.functional as F

from . import math


class Extraction:

    def __init__(self, features, logits, labels):
        self.features = features
        self.logits = logits
        self.labels = labels

class Metric:

    def __init__(self, all_all_accuracy, visible_all_accuracy, invisible_all_accuracy, visible_visible_accuracy,
                 invisible_invisible_accuracy):
        self.all_all_accuracy = all_all_accuracy
        self.visible_all_accuracy = visible_all_accuracy
        self.invisible_all_accuracy = invisible_all_accuracy
        self.visible_visible_accuracy = visible_visible_accuracy
        self.invisible_invisible_accuracy = invisible_invisible_accuracy

    PRINT_COL_WIDTH = [35, 35, 35, 35, 35]
    HEADERS = ["All-All", "Visible-All", "Invisible-All", "Visible-Visible", "Invisible-Invisible"]

    def __str__(self):
        formatted_headers = [f"{header:<{col_width}}" for header, col_width in zip(self.HEADERS, self.PRINT_COL_WIDTH)]
        data = [self.all_all_accuracy, self.visible_all_accuracy, self.invisible_all_accuracy, self.visible_visible_accuracy, self.invisible_invisible_accuracy]
        formatted_data = [f"{str(data_value):<{col_width}}" for data_value, col_width in zip(data, self.PRINT_COL_WIDTH)]
        formatted_string = f"\n{' '.join(formatted_headers)}\n{' '.join(formatted_data)}\n"
        return formatted_string


class Evaluation:

    def __init__(self, extraction, clsf_metric, nmc_metric, lp_metric):
        self.extraction = extraction
        self.clsf_metric = clsf_metric
        self.nmc_metric = nmc_metric
        self.lp_metric = lp_metric

    def __str__(self):
        return f'CLSF: {self.clsf_metric}\n NMC: {self.nmc_metric}\n LP: {self.lp_metric}\n'


def get_class_mean(domain_info, features, labels):
    all_classes = domain_info.all_classes
    num_classes = domain_info.num_classes
    dim_features = features.shape[1]
    clz_mean = torch.full((num_classes, dim_features), fill_value=-torch.inf, dtype=features.dtype,
                          device=features.device)
    for clz in all_classes:
        clz_mask = labels == clz
        if torch.sum(clz_mask) == 0:
            continue
        clz_features = features[clz_mask]
        _clz_mean = torch.mean(clz_features, dim=0)
        clz_mean[clz] = _clz_mean

    return clz_mean


def mask_similarity(similarity, row_mask, column_mask):
    similarity = copy.deepcopy(similarity)
    similarity[row_mask] = -torch.inf
    similarity[:, column_mask] = -torch.inf
    return similarity

def mask_predict_similarity(similarity, label, row_mask=None, column_mask=None):
    if row_mask is None:
        row_mask = torch.zeros(similarity.shape[0], dtype=torch.bool)
    if column_mask is None:
        column_mask = torch.zeros(similarity.shape[1], dtype=torch.bool)
    masked_similarity = mask_similarity(similarity, row_mask, column_mask)
    masked_similarity = masked_similarity[~row_mask]
    masked_label = label[~row_mask]
    accuracy = math.topk_accuracy(masked_similarity, masked_label)
    return accuracy

def generate_metric(domain_info, similarity, label):
    # Unpack domain_info
    visible_classes = domain_info.visible_classes
    invisible_classes = domain_info.invisible_classes
    dim_similarity = similarity.shape[1]
    # Masks
    visible_row_mask = torch.isin(label, visible_classes)
    invisible_row_mask = torch.isin(label, invisible_classes)
    visible_column_mask = torch.zeros(dim_similarity, dtype=torch.bool)
    visible_column_mask[visible_classes] = 1
    invisible_column_mask = torch.zeros(dim_similarity, dtype=torch.bool)
    invisible_column_mask[invisible_classes] = 1
    # Calculate metric
    #  From all classes / Over all classes
    all_all_accuracy = mask_predict_similarity(similarity, label, row_mask=None, column_mask=None)
    #  From visible classes / Over all classes
    visible_all_accuracy = mask_predict_similarity(similarity, label, row_mask=invisible_row_mask, column_mask=None)
    #  From invisible classes / Over all classes
    invisible_all_accuracy = mask_predict_similarity(similarity, label, row_mask=visible_row_mask, column_mask=None)
    #  From visible classes / Over visible classes
    visible_visible_accuracy = mask_predict_similarity(similarity, label, row_mask=invisible_row_mask, column_mask=invisible_column_mask)
    #  From invisible classes / Over invisible classes
    invisible_invisible_accuracy = mask_predict_similarity(similarity, label, row_mask=visible_row_mask, column_mask=visible_column_mask)
    # Package accuracies
    accuracies = Metric(all_all_accuracy, visible_all_accuracy, invisible_all_accuracy, visible_visible_accuracy,
                        invisible_invisible_accuracy)
    return accuracies


def evaluate_clsf(domain_info, extraction):
    logits = extraction.logits
    labels = extraction.labels
    # Calculate metric
    metric = generate_metric(domain_info, logits, labels)
    return metric

def evaluate_nmc(domain_info, extraction, oracle_extraction):
    # Unpack extraction
    features = extraction.features
    labels = extraction.labels
    # Unpack oracle_extraction
    oracle_features = oracle_extraction.features
    oracle_labels = oracle_extraction.labels
    # Calculate nmc similarity
    normed_oracle_features = F.normalize(oracle_features, dim=1)
    normed_features = F.normalize(features, dim=1)
    clz_mean = get_class_mean(domain_info, normed_oracle_features, oracle_labels)
    normed_clz_mean = F.normalize(clz_mean, dim=1)
    similarity = torch.matmul(normed_features, normed_clz_mean.T)
    # Calculate metric
    metric = generate_metric(domain_info, similarity, labels)
    return metric


def evaluate_lp():
    # TODO: Linear Probing Evaluation
    return None


def evaluate(domain_info, extraction, oracle_extraction):
    # Evaluate the features through the classifier (regular cnn model)
    clsf_metric = evaluate_clsf(domain_info, extraction)
    # Evaluate the features through the nearest mean classifier
    nmc_metric = evaluate_nmc(domain_info, extraction, oracle_extraction)
    # Evaluate the features through the linear probing
    lp_metric = evaluate_lp()
    # Package evaluation
    evaluation = Evaluation(extraction, clsf_metric, nmc_metric, lp_metric)
    return evaluation
