import torch
import os
from src.dataset import Multimodal_Datasets, PseudolabelMultimodalDataset


def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    print(f"Labeled ratio in get_data: {args.labeled_ratio}")
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = PseudolabelMultimodalDataset(args.data_path, dataset, split, args.aligned, args.labeled_ratio)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
        if not isinstance(data, PseudolabelMultimodalDataset):
            print(f"  - Converting cached data to PseudolabelMultimodalDataset")
            data = PseudolabelMultimodalDataset(args.data_path, dataset, split, args.aligned, args.labeled_ratio)
            torch.save(data, data_path)
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    save_dir = 'pre_trained_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model, os.path.join(save_dir, f'{name}.pt'))


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model
