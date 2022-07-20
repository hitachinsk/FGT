# under development
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self, device, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        self.gaussianKernel = self._gaussianKernel2dOpencv(kernel_size=5, sigma=1)
        self.gaussianKernel = np.reshape(self.gaussianKernel, (1, 1, 5, 5))  # 5 is kernel size
        self.gaussianKernel = torch.from_numpy(self.gaussianKernel).float().to(device)

    def forward(self, outputs, GTs, masks, cannyEdges):
        """
        Calculate the L1 loss in the edge regions
        edges are detected by canny operator
        Args:
            outputs: torch tensor, shape [b, c, h, w]
            GTs: torch tensor, shape [b, c, h, w]
            masks: torch tensor, shape [b, 1, h, w]
            cannyEdges: shape [b, c, h, w], 1 indicates edge regions, while 0 indicates nonedge regions
            cannyEdges should be provided by the dataloader

        Returns: edge loss between outputs and GTs

        """
        cannyEdges = self.gaussianBlur(cannyEdges)
        loss = self.loss_fn(outputs * cannyEdges * masks, GTs * cannyEdges * masks) / torch.mean(masks)
        return loss

    def gaussianBlur(self, cannyEdges, iteration=2):
        for i in range(iteration):
            cannyEdges = F.conv2d(cannyEdges, self.gaussianKernel, stride=1, padding=2)
        return cannyEdges

    def _gaussianKernel2dOpencv(self, kernel_size=5, sigma=1):
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        return np.multiply(kx, np.transpose(ky))


if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import ToTensor
    from skimage.feature import canny
    from torchvision.utils import save_image
    from skimage.color import rgb2gray

    output = Image.open('images/00001_res.jpg')
    GT = Image.open('images/img2.jpg')
    # mask = Image.open('images/mask.png')
    cannyMap = canny(rgb2gray(np.array(GT)), sigma=2).astype(np.float32)

    output = ToTensor()(output).unsqueeze(0)
    GT = ToTensor()(GT).unsqueeze(0)
    # mask = ToTensor()(mask).unsqueeze(0)
    mask = torch.ones_like(GT)
    cannyMap = ToTensor()(cannyMap).unsqueeze(0)
    output = output * 2 - 1
    GT = GT * 2 - 1

    EdgeLossLayer = EdgeLoss(2, 'cpu')
    edgeLoss, cannyPriority, edgeMap_output, edgeMap_GT, errorMap, errorMap2 = EdgeLossLayer(output, GT, mask, cannyMap)
    print(edgeLoss)
    save_image(cannyPriority, 'images/cannyPriority.jpg')
    save_image(edgeMap_output, 'images/edgeMap_output.jpg')
    save_image(edgeMap_GT, 'images/edgeMap_GT.jpg')
    save_image(edgeMap_output * cannyPriority, 'images/edgeMap_output_canny.jpg')
    save_image(edgeMap_GT * cannyPriority, 'images/edgeMap_GT_canny.jpg')
    save_image(errorMap, 'images/errorMap.jpg')
    save_image(errorMap2, 'images/errorMap2.jpg')
