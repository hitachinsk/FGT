import random
import numpy as np

class RandomMask():
    def __init__(self, videoLength, dataInfo):
        self.videoLength = videoLength
        self.imageHeight, self.imageWidth = dataInfo['image']['image_height'], \
                                            dataInfo['image']['image_width']
        self.maskHeight, self.maskWidth = dataInfo['mask']['mask_height'], \
                                          dataInfo['mask']['mask_width']
        try:
            self.maxDeltaHeight, self.maxDeltaWidth = dataInfo['mask']['max_delta_height'], \
                                                    dataInfo['mask']['max_delta_width']
        except KeyError:
            self.maxDeltaHeight, self.maxDeltaWidth = 0, 0

        try:
            self.verticalMargin, self.horizontalMargin = dataInfo['mask']['vertical_margin'], \
                                                         dataInfo['mask']['horizontal_margin']
        except KeyError:
            self.verticalMargin, self.horizontalMargin = 0, 0

    def __call__(self):
        from .utils import random_bbox
        from .utils import bbox2mask
        masks = []
        bbox = random_bbox(self.imageHeight, self.imageWidth, self.verticalMargin, self.horizontalMargin,
                           self.maskHeight, self.maskWidth)
        if random.uniform(0, 1) > 0.5:
            mask = bbox2mask(self.imageHeight, self.imageWidth, 0, 0, bbox)
            for frame in range(self.videoLength):
                masks.append(mask)
        else:
            for frame in range(self.videoLength):
                delta_h, delta_w = random.randint(-3, 3), random.randint(-3, 3)  # 每次向四个方向移动三个像素以内
                bbox = list(bbox)
                bbox[0] = min(max(self.verticalMargin, bbox[0] + delta_h), self.imageHeight - self.verticalMargin - bbox[2])
                bbox[1] = min(max(self.horizontalMargin, bbox[1] + delta_w), self.imageWidth - self.horizontalMargin - bbox[3])
                mask = bbox2mask(self.imageHeight, self.imageWidth, 0, 0, bbox)
                masks.append(mask)
        masks = np.stack(masks, axis=0)
        if len(masks.shape) == 3:
            masks = masks[:, :, :, np.newaxis]
        assert len(masks.shape) == 4, 'Wrong mask dimension {}'.format(len(masks.shape))
        return masks


class MidRandomMask():
    ### This mask is considered without random motion
    def __init__(self, videoLength, dataInfo):
        self.videoLength = videoLength
        self.imageHeight, self.imageWidth = dataInfo['image']['image_height'], \
                                            dataInfo['image']['image_width']
        self.maskHeight, self.maskWidth = dataInfo['mask']['mask_height'], \
                                          dataInfo['mask']['mask_width']

    def __call__(self):
        from .utils import mid_bbox_mask
        mask = mid_bbox_mask(self.imageHeight, self.imageWidth, self.maskHeight, self.maskWidth)
        masks = []
        for _ in range(self.videoLength):
            masks.append(mask)
        return mask


class MatrixMask():
    ### This mask is considered without random motion
    def __init__(self, videoLength, dataInfo):
        self.videoLength = videoLength
        self.imageHeight, self.imageWidth = dataInfo['image']['image_height'], \
                                            dataInfo['image']['image_width']
        self.maskHeight, self.maskWidth = dataInfo['mask']['mask_height'], \
                                          dataInfo['mask']['mask_width']
        try:
            self.row, self.column = dataInfo['mask']['row'], \
                                dataInfo['mask']['column']
        except KeyError:
            self.row, self.column = 5, 4

    def __call__(self):
        from .utils import matrix2bbox
        mask = matrix2bbox(self.imageHeight, self.imageWidth, self.maskHeight,
                           self.maskWidth, self.row, self.column)
        masks = []
        for video in range(self.videoLength):
            masks.append(mask)
        return mask


class FreeFormMask():
    def __init__(self, videoLength, dataInfo):
        self.videoLength = videoLength
        self.imageHeight, self.imageWidth = dataInfo['image']['image_height'], \
                                            dataInfo['image']['image_width']
        self.maxVertex = dataInfo['mask']['max_vertex']
        self.maxLength = dataInfo['mask']['max_length']
        self.maxBrushWidth = dataInfo['mask']['max_brush_width']
        self.maxAngle = dataInfo['mask']['max_angle']

    def __call__(self):
        from .utils import freeFormMask
        mask = freeFormMask(self.imageHeight, self.imageWidth,
                     self.maxVertex, self.maxLength,
                     self.maxBrushWidth, self.maxAngle)
        return mask


class StationaryMask():
    def __init__(self, videoLength, dataInfo):
        self.videoLength = videoLength
        self.imageHeight, self.imageWidth = dataInfo['image']['image_height'], \
                                            dataInfo['image']['image_width']
        # self.maxPointNum = dataInfo['mask']['max_point_num']
        # self.maxLength = dataInfo['mask']['max_length']

    def __call__(self):
        from .STTN_mask import create_random_shape_with_random_motion
        masks = create_random_shape_with_random_motion(self.videoLength, 0.9, 1.1, 1, 10, self.imageHeight, self.imageWidth)
        masks = np.stack(masks, axis=0)
        if len(masks.shape) == 3:
            masks = masks[:, :, :, np.newaxis]
        assert len(masks.shape) == 4, 'Your masks with a wrong shape {}'.format(len(masks.shape))
        return masks