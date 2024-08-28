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
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False,
             dropout_l=0.0, dropout_a=0.0, dropout_v=0.0):
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
        # dropout rates for each modality
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
        mask = (torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() > dropout_rate).float()
        return x * mask.unsqueeze(-1).expand_as(x)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.apply_dropout(self.text[index], self.dropout_l),
             self.apply_dropout(self.audio[index], self.dropout_a),
             self.apply_dropout(self.vision[index], self.dropout_v))
        Y = self.labels[index]

        if self.meta is None:
            META = (0, 0, 0)
        else:
            if self.data == 'mosi':
                META = (self.meta[index][0].decode('UTF-8'),
                        self.meta[index][1].decode('UTF-8'),
                        self.meta[index][2].decode('UTF-8'))
            else:
                META = (self.meta[index][0], self.meta[index][1], self.meta[index][2])

        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)

        return X, Y, META

