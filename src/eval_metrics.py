import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))

    print("-" * 50)


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()
        
        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
            test_truth_i = test_truth[:,emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()
        
        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds,axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)


def eval_semi_supervised(labeled_results, unlabeled_results, labeled_truths, threshold=0.95):
    """
    Evaluate the performance of a semi-supervised model.

    Args:
    labeled_results (np.array): Predictions for labeled data
    unlabeled_results (np.array): Predictions for unlabeled data
    labeled_truths (np.array): True labels for labeled data
    threshold (float): Confidence threshold for pseudo-labeling

    Returns:
    dict: A dictionary containing various evaluation metrics
    """
    # Metrics for labeled data
    labeled_accuracy = accuracy_score(labeled_truths, np.argmax(labeled_results, axis=1))
    labeled_f1 = f1_score(labeled_truths, np.argmax(labeled_results, axis=1), average='weighted')

    # Metrics for unlabeled data
    unlabeled_confidence = np.max(unlabeled_results, axis=1)
    high_confidence_mask = unlabeled_confidence > threshold
    high_confidence_ratio = np.mean(high_confidence_mask)

    # Pseudo-label distribution
    pseudo_labels = np.argmax(unlabeled_results, axis=1)
    unique, counts = np.unique(pseudo_labels[high_confidence_mask], return_counts=True)
    pseudo_label_distribution = dict(zip(unique, counts / sum(counts)))

    return {
        "labeled_accuracy": labeled_accuracy,
        "labeled_f1": labeled_f1,
        "unlabeled_high_confidence_ratio": high_confidence_ratio,
        "pseudo_label_distribution": pseudo_label_distribution
    }


def detailed_eval_semi_supervised(model, labeled_loader, unlabeled_loader, threshold=0.95):
    """
    Perform a detailed evaluation of the semi-supervised model.

    Args:
    model (torch.nn.Module): The trained model
    labeled_loader (DataLoader): DataLoader for labeled data
    unlabeled_loader (DataLoader): DataLoader for unlabeled data
    threshold (float): Confidence threshold for pseudo-labeling

    Returns:
    dict: A dictionary containing detailed evaluation metrics
    """
    model.eval()
    labeled_preds, labeled_truths = [], []
    unlabeled_preds = []

    with torch.no_grad():
        # Evaluate on labeled data
        for batch in labeled_loader:
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**inputs)
            labeled_preds.append(outputs.cpu().numpy())
            labeled_truths.append(batch['label'].numpy())

        # Evaluate on unlabeled data
        for batch in unlabeled_loader:
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**inputs)
            unlabeled_preds.append(outputs.cpu().numpy())

    labeled_preds = np.concatenate(labeled_preds)
    labeled_truths = np.concatenate(labeled_truths)
    unlabeled_preds = np.concatenate(unlabeled_preds)

    # Basic semi-supervised evaluation
    basic_metrics = eval_semi_supervised(labeled_preds, unlabeled_preds, labeled_truths, threshold)

    # Detailed classification report for labeled data
    class_report = classification_report(labeled_truths, np.argmax(labeled_preds, axis=1), output_dict=True)

    # Confusion matrix for labeled data
    conf_matrix = confusion_matrix(labeled_truths, np.argmax(labeled_preds, axis=1))

    return {
        **basic_metrics,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }


# Keep the existing evaluation functions (eval_mosei_senti, eval_mosi, eval_iemocap)
# ...

# Add a new function to choose the appropriate evaluation based on the dataset
def evaluate_semi_supervised(results, truths, dataset, unlabeled_results=None):
    if unlabeled_results is not None:
        return detailed_eval_semi_supervised(results, unlabeled_results, truths)
    elif dataset == "mosei_senti":
        return eval_mosei_senti(results, truths, exclude_zero=True)
    elif dataset == 'mosi':
        return eval_mosi(results, truths, exclude_zero=True)
    elif dataset == 'iemocap':
        return eval_iemocap(results, truths)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")