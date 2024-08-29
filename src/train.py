import torch
from torch import nn
import sys
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.eval_metrics import *
from src.data_augmentation import augment_data
from src.dataset import get_semi_supervised_data_loaders
from src.hooks import PseudoLabelingHook, ConsistencyRegularizationHook
from src.eval_metrics import eval_mosei_senti, eval_mosi, eval_iemocap

components = {
    'model': model,
    'pseudo_labeling_hook': PseudoLabelingHook(threshold=hyp_params.pseudo_threshold),
    'consistency_hook': ConsistencyRegularizationHook(consistency_type=hyp_params.consistency_type)
}
def initiate(hyp_params, components, labeled_loader, unlabeled_loader, valid_loader, test_loader):
    model = components['model']
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    labeled_loader, unlabeled_loader = get_semi_supervised_data_loaders(hyp_params)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, hyp_params, components, labeled_loader, unlabeled_loader, valid_loader, test_loader)

def train_model(settings, hyp_params, components, labeled_loader, unlabeled_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    pseudo_labeling_hook = components['pseudo_labeling_hook']
    consistency_hook = components['consistency_hook']

    def train(model, optimizer, criterion, epoch):
        epoch_loss = 0
        model.train()
        num_batches = min(len(labeled_loader), len(unlabeled_loader))
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for i_batch in range(num_batches):
            optimizer.zero_grad()

            # Process labeled data
            labeled_batch = next(labeled_iter)
            l_text, l_audio, l_vision = labeled_batch[0][1], labeled_batch[0][2], labeled_batch[0][3]
            labels = labeled_batch[1]

            if hyp_params.use_cuda:
                l_text, l_audio, l_vision = l_text.cuda(), l_audio.cuda(), l_vision.cuda()
                labels = labels.cuda()

            l_preds, _ = model(l_text, l_audio, l_vision)
            sup_loss = criterion(l_preds, labels)

            # Process unlabeled data
            unlabeled_batch = next(unlabeled_iter)
            u_text, u_audio, u_vision = unlabeled_batch[0][1], unlabeled_batch[0][2], unlabeled_batch[0][3]

            if hyp_params.use_cuda:
                u_text, u_audio, u_vision = u_text.cuda(), u_audio.cuda(), u_vision.cuda()

            u_preds, _ = model(u_text, u_audio, u_vision)

            # Generate pseudo-labels
            pseudo_labels, mask = pseudo_labeling_hook(u_preds)

            # Unsupervised loss
            unsup_loss = criterion(u_preds[mask], pseudo_labels[mask])

            # Consistency regularization
            if hyp_params.use_consistency:
                u_text_aug, u_audio_aug, u_vision_aug = augment_data(u_text, u_audio, u_vision)
                u_preds_aug, _ = model(u_text_aug, u_audio_aug, u_vision_aug)
                consistency_loss = consistency_hook(u_preds, u_preds_aug)
            else:
                consistency_loss = 0

            # Combine losses
            lambda_u = hyp_params.lambda_u * linear_rampup(epoch, hyp_params.warmup_epochs)
            total_loss = sup_loss + lambda_u * (unsup_loss + hyp_params.consistency_weight * consistency_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            epoch_loss += total_loss.item()

        return epoch_loss / num_batches

    def linear_rampup(current, warmup_steps):
        if warmup_steps == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, warmup_steps)
            return float(current) / warmup_steps

    def evaluate(model, criterion, data_loader):
        model.eval()
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for batch in data_loader:
                text, audio, vision = batch[0][1], batch[0][2], batch[0][3]
                labels = batch[1]

                if hyp_params.use_cuda:
                    text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()
                    labels = labels.cuda()

                preds, _ = model(text, audio, vision)
                total_loss += criterion(preds, labels).item()
                results.append(preds)
                truths.append(labels)

        avg_loss = total_loss / len(data_loader)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, criterion, epoch)
        val_loss, _, _ = evaluate(model, criterion, valid_loader)
        test_loss, _, _ = evaluate(model, criterion, test_loader)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)

        print("-" * 50)
        print(f'Epoch {epoch:2d} | Time {duration:5.4f} sec | Train Loss {train_loss:5.4f} | Valid Loss {val_loss:5.4f} | Test Loss {test_loss:5.4f}')
        print("-" * 50)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, criterion, test_loader)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    sys.stdout.flush()
    # input('[Press Any Key to start another run]')
