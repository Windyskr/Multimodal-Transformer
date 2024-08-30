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
        # 检查所有标签的维度
        dims = [label.dim() for label in labels]
        if all(dim == dims[0] for dim in dims):
            # 如果所有标签维度相同，直接堆叠
            labels = torch.stack([label.float().cpu() for label in labels])
        else:
            # 如果维度不同，先将所有标签展平为一维，然后堆叠
            labels = torch.stack([label.float().cpu().view(-1) for label in labels])
    else:
        # 如果标签不是张量，将其转换为浮点型张量
        labels = torch.tensor(labels, dtype=torch.float32)

    # 处理元数据（META）
    meta_processed = []
    for m in zip(*meta):
        if isinstance(m[0], bytes):
            meta_processed.append([item.decode('utf-8') if isinstance(item, bytes) else item for item in m])
        else:
            meta_processed.append(m)

    # 处理 is_labeled
    is_labeled = torch.tensor(is_labeled, dtype=torch.bool)

    return data_processed, labels, meta_processed, is_labeled