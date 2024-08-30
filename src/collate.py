# src/collate.py

import torch


def custom_collate(batch):
    # 分离数据、标签、元数据和是否有标签
    data, labels, meta, is_labeled = zip(*batch)

    # 处理数据（X）
    data_processed = []
    for d in zip(*data):
        if isinstance(d[0], torch.Tensor):
            data_processed.append(torch.stack(d))
        else:
            data_processed.append(d)

    # 处理标签（Y）
    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack([label.cpu() for label in labels])  # 确保所有标签都在 CPU 上

    # 处理元数据（META）
    meta_processed = []
    for m in zip(*meta):
        if isinstance(m[0], bytes):
            meta_processed.append([item.decode('utf-8') if isinstance(item, bytes) else item for item in m])
        else:
            meta_processed.append(m)

    # 处理 is_labeled
    is_labeled = torch.tensor(is_labeled)

    return data_processed, labels, meta_processed, is_labeled