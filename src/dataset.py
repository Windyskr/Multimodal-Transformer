import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data
        
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META        

class PseudolabelMultimodalDataset(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False, labeled_ratio=0.5):
        super(PseudolabelMultimodalDataset, self).__init__()
        self.dataset_path = dataset_path
        self.data = data
        self.split_type = split_type
        self.if_align = if_align
        self.labeled_ratio = float(labeled_ratio)  # Ensure it's a float

        print(f"Initializing PseudolabelMultimodalDataset with labeled_ratio: {self.labeled_ratio}")

        self.load_data()
        self.process_data()

    def load_data(self):
        dataset_path = os.path.join(self.dataset_path,
                                    self.data + '_data.pkl' if self.if_align else self.data + '_data_noalign.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))

        self.vision = torch.tensor(dataset[self.split_type]['vision'].astype(np.float32)).cpu()
        self.text = torch.tensor(dataset[self.split_type]['text'].astype(np.float32)).cpu()
        self.audio = dataset[self.split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu()
        self.labels = torch.tensor(dataset[self.split_type]['labels'].astype(np.float32)).cpu()

        self.meta = dataset[self.split_type]['id'] if 'id' in dataset[self.split_type].keys() else None

    def process_data(self):
        if self.split_type == 'train':
            # Ensure all tensors are on CPU and the correct type
            self.vision = self.vision.cpu().float()
            self.text = self.text.cpu().float()
            self.audio = self.audio.cpu().float()
            self.labels = self.labels.cpu().float()

            # Shuffle the data
            perm = torch.randperm(len(self.labels))
            self.vision = self.vision[perm]
            self.text = self.text[perm]
            self.audio = self.audio[perm]
            self.labels = self.labels[perm]
            if self.meta is not None:
                self.meta = [self.meta[i] for i in perm]  # Assuming meta is a list

            # Split into labeled and unlabeled
            n_labeled = int(len(self.labels) * self.labeled_ratio)
            self.labeled_mask = torch.zeros(len(self.labels), dtype=torch.bool)
            self.labeled_mask[:n_labeled] = True

            print(f"Labeled mask dtype: {self.labeled_mask.dtype}")
            print(f"Labeled mask shape: {self.labeled_mask.shape}")
            print(f"Labeled mask sum: {self.labeled_mask.int().sum().item()}")

            # For unlabeled data, set labels to -1
            unlabeled_mask = ~self.labeled_mask  # 使用 ~ 操作符进行逻辑非操作
            self.labels[unlabeled_mask] = -1

            print(f"Labels dtype: {self.labels.dtype}")
            print(f"Labels shape: {self.labels.shape}")
            print(f"Number of -1 labels: {(self.labels == -1).sum().item()}")

        else:
            # For validation and test sets, all data is labeled
            self.labeled_mask = torch.ones(len(self.labels), dtype=torch.bool)

        # Ensure labeled_mask is a boolean tensor
        self.labeled_mask = self.labeled_mask.bool()

    def update_pseudolabels(self, indices, new_labels):
        """
        Update the labels for unlabeled data with new pseudo-labels.

        Args:
            indices (list): Indices of the samples to update
            new_labels (torch.Tensor): New pseudo-labels for the samples
        """
        for idx, new_label in zip(indices, new_labels):
            if not self.labeled_mask[idx]:
                self.labels[idx] = new_label

    def get_unlabeled_indices(self):
        """
        Return the indices of unlabeled samples.
        """
        return torch.where(~self.labeled_mask)[0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0, 0, 0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'),
                    self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META, self.labeled_mask[index].item()