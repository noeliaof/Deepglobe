import torch
import numpy as np
import torch.nn as nn

def get_accuracy(ground_truths, predictions):
    accuracy = 0
    for i in range(len(predictions)):
        c, w, h = predictions[i].size()
        total_pixels = w * h
        prediction = predictions[i]
        truth = ground_truths[i]
        output = torch.argmax(prediction, dim=0)
        iteration_accuracy = (torch.sum((output == truth))).to(dtype=torch.float) / total_pixels
        accuracy += iteration_accuracy
        accuracy /= len(predictions)
    return accuracy

def get_f1_score( ground_truths, predictions):
    precision_arr = []
    recall_arr = []
    f1_arr = []
    for i in range(len(predictions)):
        c, w, h = predictions[i].size()
        prediction = predictions[i]
        truth = ground_truths[i]
        output = torch.argmax(prediction, dim=0)

        tp = (truth * output).sum()
        tn = ((1 - truth) * (1 - output)).sum()
        fp = ((1 - truth) * output).sum()
        fn = (truth * (1 - output)).sum()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

        precision_arr.append(precision)
        recall_arr.append(recall)
        f1_arr.append(f1_score)

    return np.average(precision_arr), np.average(recall_arr), np.average(f1_arr)


def get_iou(ground_truths, predictions):
    mean_iou = 0
    for i in range(len(predictions)):
        prediction = predictions[i].view(-1)
        truth = ground_truths[i].view(-1)

        intersection = (prediction * truth).sum()
        total = (prediction + truth).sum()
        union = total - intersection

        iou = intersection / union
        mean_iou += iou

    mean_iou /= len(predictions)
    return mean_iou


class WeightedDiceLoss(nn.Module):
    def __init__(self, num_classes, weights=None, smooth=1.0):
        super(WeightedDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.smooth = smooth
        self.__name__ = 'WeightedDiceLoss'

    def forward(self, logits, targets):
        # Calculate true positive, false positive, and false negative
        intersection = (logits * targets).sum(dim=(2, 3))
        true_positive = 2.0 * intersection + self.smooth
        false_positive = logits.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth

        # Calculate class weights (if provided)
        if self.weights is not None:
            weights = self.weights.to(targets.device)
            weights = weights.view(1, -1, 1, 1)
            true_positive = true_positive * weights
            false_positive = false_positive * weights

        # Calculate Dice loss for each class
        dice_loss = 1.0 - (true_positive / false_positive).mean()

        return dice_loss