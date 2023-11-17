import torch
import torch.nn as nn
from collections import OrderedDict

def is_tensor(x):
    return isinstance(x, torch.Tensor)

class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v

class TrainLoss(nn.Module):
    def __init__(self, cls_weight=0.5, avg_modality = True):
        """
        Initialize the class.
        Args:
            cls_weight: the weight of the classification loss term.
        """
        super(TrainLoss, self).__init__()
        self.cls_weight = cls_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.info = Odict()
        self.avg = avg_modality

    def forward(self, cls_logits_list, m_labels, cls_labels_list, preds, labels):
        """
        Calculate the weighted sum of bce and mse losses.
        Args:
            cls_logits_list: A list of tensors for classification task, each with shape [n, c]
            m_labels: A binary mask for modality presence [n, 4]
            cls_labels_list: A list of true labels for classification task, each with shape [n, c]
            preds: Predictions for regression task [n, 1]
            labels: True labels for regression task [n, 1]
        Returns:
            A tuple of the total loss and a dictionary with loss details.
        """
        bce_total_loss = 0.0
        # Iterate over each modality
        for idx, (cls_logits, cls_labels) in enumerate(zip(cls_logits_list, cls_labels_list)):
            # Compute BCE loss for the current modality
            bce_loss = self.bce_loss(cls_logits, cls_labels)
            # Apply modality mask, idx+1 to skip text
            modality_mask = m_labels[:, idx+1].unsqueeze(1)  # Make it [n, 1] to match the loss shape
            bce_loss = bce_loss * modality_mask
            # Sum over the batch for the current modality and average by the number of present modalities
            bce_total_loss += bce_loss.sum() / modality_mask.sum()
        
        if self.avg:
            # Average the bce loss over the number of modalities, optional
            bce_total_loss /= len(cls_logits_list)

        # Compute MSE loss
        mse_loss = self.mse_loss(preds, labels)

        # Weighted sum of the losses
        total_loss = self.cls_weight * bce_total_loss + mse_loss

        # Update the information dictionary
        self.info = {
            'total_loss': total_loss.item(),
            'bce_loss': bce_total_loss.item() if is_tensor(bce_loss) else bce_total_loss,
            'mse_loss': mse_loss.item()
        }

        return total_loss, self.info
