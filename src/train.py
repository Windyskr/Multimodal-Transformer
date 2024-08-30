# src/train.py

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


def initiate(hyp_params, labeled_loader, unlabeled_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model)(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

    settings = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler
    }

    return train_model(settings, hyp_params, labeled_loader, unlabeled_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, labeled_loader, unlabeled_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion, epoch):
        model.train()
        total_loss = 0.0

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for i_batch in range(hyp_params.n_train // (2 * hyp_params.batch_size)):
            # Labeled data
            try:
                l_data, l_labels, l_meta, _ = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                l_data, l_labels, l_meta, _ = next(labeled_iter)

            sample_ind, l_text, l_audio, l_vision = l_data
            l_text = l_text.cuda() if hyp_params.use_cuda else l_text
            l_audio = l_audio.cuda() if hyp_params.use_cuda else l_audio
            l_vision = l_vision.cuda() if hyp_params.use_cuda else l_vision
            l_labels = l_labels.cuda() if hyp_params.use_cuda else l_labels

            # Unlabeled data
            try:
                u_data, _, u_meta, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                u_data, _, u_meta, _ = next(unlabeled_iter)

            u_sample_ind, u_text, u_audio, u_vision = u_data
            u_text = u_text.cuda() if hyp_params.use_cuda else u_text
            u_audio = u_audio.cuda() if hyp_params.use_cuda else u_audio
            u_vision = u_vision.cuda() if hyp_params.use_cuda else u_vision

            # Forward pass
            l_preds, l_hiddens = model(l_text, l_audio, l_vision)
            u_preds, u_hiddens = model(u_text, u_audio, u_vision)

            # Supervised loss
            sup_loss = criterion(l_preds, l_labels)

            # Unsupervised loss (pseudo-labeling)
            pseudo_labels = u_preds.detach()
            mask = (pseudo_labels.max(1)[0] >= hyp_params.pseudo_threshold).float()
            unsup_loss = (criterion(u_preds, pseudo_labels.argmax(dim=1)) * mask).mean()

            # Consistency regularization
            u_preds_aug, _ = model(u_text, u_audio, u_vision)  # Another forward pass with different dropout
            consistency_loss = nn.MSELoss()(u_preds, u_preds_aug)

            # Total loss
            loss = sup_loss + hyp_params.unsup_weight * unsup_loss + hyp_params.consistency_weight * consistency_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / (hyp_params.n_train // (2 * hyp_params.batch_size))

    def evaluate(model, criterion, data_loader, dataset):
        model.eval()
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_data, batch_labels, _, _) in enumerate(data_loader):
                sample_ind, text, audio, vision = batch_data
                text = text.cuda() if hyp_params.use_cuda else text
                audio = audio.cuda() if hyp_params.use_cuda else audio
                vision = vision.cuda() if hyp_params.use_cuda else vision
                labels = batch_labels.cuda() if hyp_params.use_cuda else batch_labels()

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
        val_loss, _, _ = evaluate(model, criterion, valid_loader, 'valid')
        test_loss, _, _ = evaluate(model, criterion, test_loader, 'test')

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 50)
        print(
            'Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch,
                                                                                                                  duration,
                                                                                                                  train_loss,
                                                                                                                  val_loss,
                                                                                                                  test_loss))
        print("-" * 50)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, criterion, test_loader, 'test')

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    sys.stdout.flush()
    # input('[Press Any Key to start another run]')