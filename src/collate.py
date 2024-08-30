# collate.py
import torch

def custom_collate(batch):
    # 分离数据、标签和元数据
    data, labels, meta = zip(*batch)

    # 处理数据（X）
    data_processed = []
    for d in zip(*data):
        if isinstance(d[0], torch.Tensor):
            data_processed.append(torch.stack(d))
        else:
            data_processed.append(d)

    # 处理标签（Y）
    labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else labels

    # 处理元数据（META）
    meta_processed = []
    for m in zip(*meta):
        if isinstance(m[0], bytes):
            meta_processed.append([item.decode('utf-8') if isinstance(item, bytes) else item for item in m])
        else:
            meta_processed.append(m)

    return data_processed, labels, meta_processed