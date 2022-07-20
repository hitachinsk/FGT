import torch
import torch.nn as nn
import torch.nn.functional as F


def edgeLoss(preds_edges, edges):
    """

    Args:
        preds_edges: with shape [b, c, h , w]
        edges: with shape [b, c, h, w]

    Returns: Edge losses

    """
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()
    num_neg = c * h * w - num_pos
    neg_weights = (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    pos_weights = (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    weight = neg_weights * mask + pos_weights * (1 - mask)  # weight for debug
    losses = F.binary_cross_entropy_with_logits(preds_edges.float(), edges.float(), weight=weight, reduction='none')
    loss = torch.mean(losses)
    return loss


class EdgeAcc(nn.Module):
    """
    Measure the accuracy of the edge map
    """

    def __init__(self, threshold=0.5):
        super(EdgeAcc, self).__init__()
        self.threshold = threshold

    def __call__(self, pred_edge, gt_edge):
        """

        Args:
            pred_edge: Predicted edges, with shape [b, c, h, w]
            gt_edge: GT edges, with shape [b, c, h, w]

        Returns: The prediction accuracy and the recall of the edges

        """
        labels = gt_edge > self.threshold
        preds = pred_edge > self.threshold

        relevant = torch.sum(labels.float())
        selected = torch.sum(preds.float())

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((preds == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)
        return precision, recall


if __name__ == '__main__':
    edge = torch.zeros([2, 1, 10, 10])  # [b, 1, h, w] -> the extracted edges
    edge[0, :, 2:8, 2:8] = 1
    edge[1, :, 3:7, 3:7] = 1
    mask = (edge > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()
    num_neg = c * h * w - num_pos
    print(num_pos, num_neg)
    n = num_neg / (num_pos + num_neg)
    p = num_pos / (num_pos + num_neg)
    n = n.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    p = p.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    print(n * mask + p * (1 - mask))
    # weight = num_neg / (num_pos + num_neg) * mask + num_pos / (num_pos + num_neg) * (1 - mask)
    # print(weight)
