import torch
import torch.nn.functional as F

class PseudoLabelingHook:
    def __init__(self, threshold=0.95, temp=1.0):
        """
        Initialize the PseudoLabelingHook.

        Args:
        threshold (float): Confidence threshold for pseudo-labeling
        temp (float): Temperature for sharpening predictions
        """
        self.threshold = threshold
        self.temp = temp

    def __call__(self, logits):
        """
        Generate pseudo-labels from model predictions.

        Args:
        logits (torch.Tensor): Raw model outputs

        Returns:
        tuple: (pseudo_labels, mask)
        """
        with torch.no_grad():
            probs = F.softmax(logits / self.temp, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask = max_probs > self.threshold
        return pseudo_labels, mask

    def update_threshold(self, new_threshold):
        """Update the confidence threshold."""
        self.threshold = new_threshold

    def update_temperature(self, new_temp):
        """Update the temperature for sharpening."""
        self.temp = new_temp


class FixedThresholdingHook:
    def __init__(self, threshold=0.95):
        """
        Initialize the FixedThresholdingHook.

        Args:
        threshold (float): Fixed threshold for confidence
        """
        self.threshold = threshold

    def __call__(self, logits):
        """
        Apply fixed thresholding to model predictions.

        Args:
        logits (torch.Tensor): Raw model outputs

        Returns:
        torch.Tensor: Boolean mask indicating high-confidence predictions
        """
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            mask = max_probs > self.threshold
        return mask

    def update_threshold(self, new_threshold):
        """Update the fixed threshold."""
        self.threshold = new_threshold


class ConsistencyRegularizationHook:
    def __init__(self, consistency_type='mse'):
        """
        Initialize the ConsistencyRegularizationHook.

        Args:
        consistency_type (str): Type of consistency loss ('mse' or 'kl')
        """
        self.consistency_type = consistency_type

    def __call__(self, pred1, pred2):
        """
        Compute consistency loss between two predictions.

        Args:
        pred1 (torch.Tensor): First set of predictions
        pred2 (torch.Tensor): Second set of predictions

        Returns:
        torch.Tensor: Consistency loss
        """
        if self.consistency_type == 'mse':
            return F.mse_loss(pred1, pred2, reduction='mean')
        elif self.consistency_type == 'kl':
            return F.kl_div(F.log_softmax(pred1, dim=1), F.softmax(pred2, dim=1), reduction='batchmean')
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")

    def update_consistency_type(self, new_type):
        """Update the type of consistency loss."""
        if new_type in ['mse', 'kl']:
            self.consistency_type = new_type
        else:
            raise ValueError(f"Unsupported consistency type: {new_type}")