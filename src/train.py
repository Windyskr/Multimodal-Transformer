import torch
from torch import nn
import sys
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def get_CTC_module(hyp_params):
    a2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    if hyp_params.aligned or hyp_params.model=='MULT':
        ctc_criterion = None
        ctc_a2l_module, ctc_v2l_module = None, None
        ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    else:
        from warpctc_pytorch import CTCLoss
        ctc_criterion = CTCLoss()
        ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
        if hyp_params.use_cuda:
            ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.cuda(), ctc_v2l_module.cuda()
        ctc_a2l_optimizer = getattr(optim, hyp_params.optim)(ctc_a2l_module.parameters(), lr=hyp_params.lr)
        ctc_v2l_optimizer = getattr(optim, hyp_params.optim)(ctc_v2l_module.parameters(), lr=hyp_params.lr)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    
    ctc_a2l_module = settings['ctc_a2l_module']
    ctc_v2l_module = settings['ctc_v2l_module']
    ctc_a2l_optimizer = settings['ctc_a2l_optimizer']
    ctc_v2l_optimizer = settings['ctc_v2l_optimizer']
    ctc_criterion = settings['ctc_criterion']
    
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer,
              ctc_criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, batch_META, batch_MASK) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)  # if num of labels is 1
            labeled_mask = batch_MASK

            model.zero_grad()
            if ctc_criterion is not None:
                ctc_a2l_module.zero_grad()
                ctc_v2l_module.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr, labeled_mask = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), labeled_mask.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk

            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 10 else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                vision_chunks = vision.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                labeled_mask_chunks = labeled_mask.chunk(batch_chunk, dim=0)

                for i in range(batch_chunk):
                    text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    labeled_mask_i = labeled_mask_chunks[i]
                    preds_i, confidence_i, _ = net(text_i, audio_i, vision_i)

                    if hyp_params.dataset == 'iemocap':
                        preds_i = preds_i.view(-1, 2)
                        eval_attr_i = eval_attr_i.view(-1)

                    # Calculate loss for labeled data
                    labeled_loss = criterion(preds_i[labeled_mask_i], eval_attr_i[labeled_mask_i])

                    # Generate pseudo-labels for unlabeled data
                    with torch.no_grad():
                        pseudo_labels = preds_i[~labeled_mask_i].detach()
                        mask = confidence_i[~labeled_mask_i] > hyp_params.pseudolabel_threshold
                        if hyp_params.dataset in ['mosi', 'mosei']:
                            # For regression tasks, use the predictions directly
                            pseudo_labeled_loss = criterion(preds_i[~labeled_mask_i][mask], pseudo_labels[mask])
                        else:
                            # For classification tasks, use argmax
                            pseudo_labeled_loss = criterion(preds_i[~labeled_mask_i][mask],
                                                            pseudo_labels[mask].argmax(dim=-1))

                    # Combine losses
                    raw_loss = labeled_loss + hyp_params.lambda_u * pseudo_labeled_loss
                    combined_loss += raw_loss

                    # Check for NaN
                    if torch.isnan(combined_loss):
                        print(f"NaN detected in loss calculation. Batch: {i_batch}, Chunk: {i}")
                        print(f"Labeled loss: {labeled_loss}, Pseudo-labeled loss: {pseudo_labeled_loss}")
                        continue

                    combined_loss.backward()

                combined_loss = raw_loss
            else:
                preds, confidence, _ = net(text, audio, vision, temperature=2.0)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)

                # Calculate loss for labeled data
                labeled_loss = criterion(preds[labeled_mask], eval_attr[labeled_mask])

                # Generate pseudo-labels for unlabeled data
                with torch.no_grad():
                    pseudo_labels = preds[~labeled_mask].detach()
                    mask = confidence[~labeled_mask] > hyp_params.pseudolabel_threshold
                    if hyp_params.dataset in ['mosi', 'mosei']:
                        # For regression tasks, use the predictions directly
                        pseudo_labeled_loss = criterion(preds[~labeled_mask][mask], pseudo_labels[mask])
                    else:
                        # For classification tasks, use argmax
                        pseudo_labeled_loss = criterion(preds[~labeled_mask][mask], pseudo_labels[mask].argmax(dim=-1))

                # Combine losses
                raw_loss = labeled_loss + hyp_params.lambda_u * pseudo_labeled_loss
                combined_loss = raw_loss

                # Check for NaN
                if torch.isnan(combined_loss):
                    print(f"NaN detected in loss calculation. Batch: {i_batch}")
                    print(f"Labeled loss: {labeled_loss}, Pseudo-labeled loss: {pseudo_labeled_loss}")
                    continue

                combined_loss.backward()

            if ctc_criterion is not None:
                torch.nn.utils.clip_grad_norm_(ctc_a2l_module.parameters(), hyp_params.clip)
                torch.nn.utils.clip_grad_norm_(ctc_v2l_module.parameters(), hyp_params.clip)
                ctc_a2l_optimizer.step()
                ctc_v2l_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += combined_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()

            # Check for NaN in gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN detected in gradients for parameter: {name}")

        return epoch_loss / hyp_params.n_train

    def evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META, batch_MASK) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1
                labeled_mask = batch_MASK

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr, labeled_mask = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), labeled_mask.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()

                batch_size = text.size(0)

                if (ctc_a2l_module is not None) and (ctc_v2l_module is not None):
                    ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                    ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module
                    audio, _ = ctc_a2l_net(audio)  # audio aligned to text
                    vision, _ = ctc_v2l_net(vision)  # vision aligned to text

                net = nn.DataParallel(model) if batch_size > 10 else model
                outputs = net(text, audio, vision)
                preds = outputs[0]
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)

                total_loss += criterion(preds[labeled_mask], eval_attr[labeled_mask]).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        model.train()
        try:
            train_loss = train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer,
                               ctc_v2l_optimizer, ctc_criterion)
        except RuntimeError as e:
            print(f"RuntimeError in epoch {epoch}: {e}")
            print("Skipping this epoch")
            continue

        # Update pseudo-labels
        if epoch % hyp_params.pseudolabel_update_interval == 0:
            update_pseudolabels(model, train_loader, hyp_params)

        # Evaluate on validation set
        val_loss, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False)

        # Evaluate on test set
        test_loss, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)

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
    _, results, truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')

def update_pseudolabels(model, train_loader, hyp_params):
    model.eval()
    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, batch_META, batch_MASK) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            labeled_mask = batch_MASK

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()

            net = nn.DataParallel(model) if text.size(0) > 10 else model
            preds, _ = net(text, audio, vision)

            # Generate new pseudo-labels for unlabeled data
            unlabeled_preds = preds[~labeled_mask]
            confidence, new_labels = torch.max(F.softmax(unlabeled_preds, dim=1), dim=1)
            mask = confidence > hyp_params.pseudolabel_threshold

            # Update the dataset with new pseudo-labels
            train_loader.dataset.update_pseudolabels(sample_ind[~labeled_mask][mask], new_labels[mask])

    model.train()