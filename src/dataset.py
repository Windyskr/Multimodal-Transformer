import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pickle
import os
from scipy import signal
import torch
from src.data_augmentation import augment_data

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False,
                 dropout_l=0.0, dropout_a=0.0, dropout_v=0.0):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data + '_data.pkl' if if_align else data + '_data_noalign.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))

        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.data = data
        self.n_modalities = 3

        self.dropout_l = dropout_l
        self.dropout_a = dropout_a
        self.dropout_v = dropout_v

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]

    def apply_dropout(self, x, dropout_rate):
        if dropout_rate == 0:
            return x
        mask = torch.bernoulli(torch.full(x.shape, 1 - dropout_rate, device=x.device, dtype=x.dtype))
        return x * mask
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.apply_dropout(self.text[index], self.dropout_l),
             self.apply_dropout(self.audio[index], self.dropout_a),
             self.apply_dropout(self.vision[index], self.dropout_v))
        Y = self.labels[index]
        META = self.meta[index] if self.meta is not None else (0, 0, 0)

        return X, Y, META


class SemiSupervisedMultimodalDataset(Multimodal_Datasets):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False,
                 labeled_ratio=0.1, dropout_l=0.0, dropout_a=0.0, dropout_v=0.0):
        super().__init__(dataset_path, data, split_type, if_align, dropout_l, dropout_a, dropout_v)

        self.labeled_ratio = labeled_ratio
        self.split_labeled_unlabeled()

    def split_labeled_unlabeled(self):
        total_samples = len(self.labels)
        labeled_samples = int(total_samples * self.labeled_ratio)

        # 随机选择有标签样本
        self.labeled_indices = np.random.choice(total_samples, labeled_samples, replace=False)
        self.unlabeled_indices = np.setdiff1d(np.arange(total_samples), self.labeled_indices)

    def __getitem__(self, index):
        X, Y, META = super().__getitem__(index)

        if index in self.labeled_indices:
            is_labeled = True
        else:
            is_labeled = False
            Y = torch.tensor(-1)  # 为无标签数据使用占位符标签

        return X, Y, META, is_labeled


# Function to create data loaders
def get_semi_supervised_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}_semi.dt'

    if not os.path.exists(data_path):
        print(f"  - Creating new semi-supervised {split} data")
        data = SemiSupervisedMultimodalDataset(args.data_path, dataset, split, args.aligned,
                                               args.labeled_ratio, args.dropout_l, args.dropout_a, args.dropout_v)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached semi-supervised {split} data")
        data = torch.load(data_path)
        data.dropout_l = args.dropout_l
        data.dropout_a = args.dropout_a
        data.dropout_v = args.dropout_v

    return data

def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data