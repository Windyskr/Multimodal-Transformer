import torch
import os
from src.dataset import Multimodal_Datasets


def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned,
                                   args.dropout_l, args.dropout_a, args.dropout_v)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
        data.dropout_l = args.dropout_l
        data.dropout_a = args.dropout_a
        data.dropout_v = args.dropout_v
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model

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