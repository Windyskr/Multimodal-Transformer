import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
from src.dataset import get_semi_supervised_data_loaders, get_data


parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')

# add dropout rate for each modality
parser.add_argument('--dropout_l', type=float, default=0.0,
                    help='dropout rate for language modality')
parser.add_argument('--dropout_a', type=float, default=0.0,
                    help='dropout rate for audio modality')
parser.add_argument('--dropout_v', type=float, default=0.0,
                    help='dropout rate for visual modality')
# Semi-supervised learning
parser.add_argument('--labeled_ratio', type=float, default=0.1, help='Ratio of labeled data to use')
parser.add_argument('--pseudo_threshold', type=float, default=0.95, help='Confidence threshold for pseudo-labeling')
parser.add_argument('--consistency_type', type=str, default='mse', choices=['mse', 'kl'], help='Type of consistency loss')
parser.add_argument('--consistency_weight', type=float, default=1.0, help='Weight for consistency loss')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sharpening predictions')
parser.add_argument('--unsup_warm_up', type=float, default=0.4, help='Warm-up ratio for unsupervised loss')
parser.add_argument('--ema_decay', type=float, default=0.999, help='Decay rate for EMA model')
parser.add_argument('--use_mixup', action='store_true', help='Whether to use mixup augmentation')
parser.add_argument('--mixup_alpha', type=float, default=0.4, help='Alpha parameter for mixup')

# Arguments for training control
parser.add_argument('--use_warmup', action='store_true', help='Whether to use learning rate warmup')
parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
parser.add_argument('--use_ema', action='store_true', help='Whether to use EMA model averaging')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

labeled_loader, unlabeled_loader = get_semi_supervised_data_loaders(args)
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
   
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
args.orig_d_l, args.orig_d_a, args.orig_d_v = labeled_loader.dataset.get_dim()
args.l_len, args.a_len, args.v_len = labeled_loader.dataset.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train = len(labeled_loader.dataset) + len(unlabeled_loader.dataset)
hyp_params.n_labeled = len(labeled_loader.dataset)
hyp_params.n_unlabeled = len(unlabeled_loader.dataset)
hyp_params.n_valid = len(valid_data)
hyp_params.n_test = len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')
print(f"Labeled data: {args.n_labeled}, Unlabeled data: {args.n_unlabeled}")

if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, labeled_loader, unlabeled_loader, valid_loader, test_loader)

