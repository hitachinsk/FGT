import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelLayer(nn.Module):
    def __init__(self, device):
        super(SobelLayer, self).__init__()
        self.kernel_x = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0) / 4.
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1.]]).unsqueeze(0).unsqueeze(0) / 4.
        self.kernel_x = self.kernel_x.to(device)
        self.kernel_y = self.kernel_y.to(device)
        self.pad = nn.ReplicationPad2d(padding=1)
        self.absLayer = nn.ReLU()

    def forward(self, images):
        """

        Args:
            images: images with shape [b, c, h, w]

        Returns:

        """
        images = self.pad(images)
        gray_images = self._convertGrey(images)

        edge_x = F.conv2d(gray_images, self.kernel_x, stride=1)
        edge_y = F.conv2d(gray_images, self.kernel_y, stride=1)
        edge = (self.absLayer(edge_x) + self.absLayer(edge_y)) / 2
        return edge

    def _convertGrey(self, image):
        """
        grey = 0.299 * r + 0.587 * g + 0.110 * b
        Args:
            image: RGB image

        Returns: Grey scale image

        """
        grey_image = image[:, 0] * 0.299 + image[:, 1] * 0.587 + 0.110 * image[:, 2]
        output = grey_image.unsqueeze(1)
        return output


class SeperateSobelLayer(nn.Module):
    def __init__(self, device):
        super(SeperateSobelLayer, self).__init__()
        self.kernel_x = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1.]]).unsqueeze(0).unsqueeze(0)
        self.weight = torch.zeros([6, 3, 3, 3])
        for c in range(3):
            self.weight[2 * c, c] = self.kernel_x
            self.weight[2 * c + 1, c] = self.kernel_y
        self.weight = self.weight.to(device)

    def forward(self, images):
        """

        Args:
            images: with shape [b, c, h, w]

        Returns: sobel gradient image with shape [b, c, h, w] (same padding)

        """
        gradientMap = F.conv2d(images, self.weight, stride=1, padding=1)
        return gradientMap
