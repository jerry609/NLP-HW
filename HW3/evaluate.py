from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import torch


def evaluate(model, data_loader, config, compute_metrics=True):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for texts, labels, lengths in data_loader:
            texts = texts.to(config.device)
            labels = labels.to(config.device)

            # 创建mask
            mask = torch.zeros(texts.shape, dtype=torch.bool, device=config.device)
            for idx, length in enumerate(lengths):
                mask[idx, :length] = 1

            # 前向传播
            emissions = model(texts, mask)
            pred = model.crf.decode(emissions, mask=mask)

            # 收集预测和真实标签
            for i, length in enumerate(lengths):
                predictions.append([config.idx2tag[p] for p in pred[i][:length]])
                true_labels.append([config.idx2tag[l.item()] for l in labels[i][:length]])

    if compute_metrics:
        return compute_metrics_with_and_without_o(true_labels, predictions)
    return predictions


def compute_metrics_with_and_without_o(true_labels, pred_labels):
    # 计算包含O的指标
    metrics_with_o = classification_report(true_labels, pred_labels, output_dict=True)

    # 计算不包含O的指标
    true_no_o = [[t for t in seq if t != 'O'] for seq in true_labels]
    pred_no_o = [[p for p in seq if p != 'O'] for seq in pred_labels]
    metrics_no_o = classification_report(true_no_o, pred_no_o, output_dict=True)

    return {
        'with_o': metrics_with_o,
        'without_o': metrics_no_o
    }
