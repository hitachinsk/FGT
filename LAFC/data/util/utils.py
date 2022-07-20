import random
import numpy as np
import cv2

def random_bbox(img_height, img_width, vertical_margin, horizontal_margin, mask_height, mask_width):
    maxt = img_height - vertical_margin - mask_height
    maxl = img_width - horizontal_margin - mask_width

    t = random.randint(vertical_margin, maxt)
    l = random.randint(horizontal_margin, maxl)
    h = random.randint(mask_height // 2, mask_height)
    w = random.randint(mask_width // 2, mask_width)
    return (t, l, h, w)  # 产生随机块状box,这个box后面会发展成为mask


def mid_bbox_mask(img_height, img_width, mask_height, mask_width):
    def npmask(bbox, height, width):
        mask = np.zeros((height, width, 1), np.float32)
        mask[bbox[0]: bbox[0] + bbox[2], bbox[1]: bbox[1] + bbox[3], :] = 255.
        return mask

    bbox = (img_height * 3 // 8, img_width * 3 // 8, mask_height, mask_width)
    mask = npmask(bbox, img_height, img_width)

    return mask


def bbox2mask(img_height, img_width, max_delta_height, max_delta_width, bbox):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [B, 1, H, W]

    """

    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((height, width, 1), np.float32)
        h = np.random.randint(delta_h // 2 + 1)  # 防止有0产生
        w = np.random.randint(delta_w // 2 + 1)
        mask[bbox[0] + h: bbox[0] + bbox[2] - h, bbox[1] + w: bbox[1] + bbox[3] - w, :] = 255.  # height_true = height - 2 * h, width_true = width - 2 * w
        return mask

    mask = npmask(bbox, img_height, img_width,
                  max_delta_height,
                  max_delta_width)

    return mask


def matrix2bbox(img_height, img_width, mask_height, mask_width, row, column):
    """Generate masks with a matrix form
    @param img_height
    @param img_width
    @param mask_height
    @param mask_width
    @param row: number of blocks in row
    @param column: number of blocks in column
    @return mbbox: multiple bboxes in (y, h, h, w) manner
    """
    assert img_height - column * mask_height > img_height // 2, "Too many masks across a column"
    assert img_width - row * mask_width > img_width // 2, "Too many masks across a row"

    interval_height = (img_height - column * mask_height) // (column + 1)
    interval_width = (img_width - row * mask_width) // (row + 1)

    mbbox = []
    for i in range(row):
        for j in range(column):
            y = interval_height * (j+1) + j * mask_height
            x = interval_width * (i+1) + i * mask_width
            mbbox.append((y, x, mask_height, mask_width))
    return mbbox


def mbbox2masks(img_height, img_width, mbbox):

    def npmask(mbbox, height, width):
        mask = np.zeros((height, width, 1), np.float32)
        for bbox in mbbox:
            mask[bbox[0]: bbox[0] + bbox[2], bbox[1]: bbox[1] + bbox[3], :] = 255.  # height_true = height - 2 * h, width_true = width - 2 * w
        return mask

    mask = npmask(mbbox, img_height, img_width)

    return mask


def draw_line(mask, startX, startY, angle, length, brushWidth):
    """assume the size of mask is (H,W,1)
    """
    assert len(mask.shape) == 2 or mask.shape[2] == 1, "The channel of mask doesn't fit the opencv format"
    offsetX = int(np.round(length * np.cos(angle)))
    offsetY = int(np.round(length * np.sin(angle)))
    endX = startX + offsetX
    endY = startY + offsetY
    if endX > mask.shape[1]:
        endX = mask.shape[1]
    if endY > mask.shape[0]:
        endY = mask.shape[0]
    mask_processed = cv2.line(mask, (startX, startY), (endX, endY), 255, brushWidth)
    return mask_processed, endX, endY


def draw_circle(mask, circle_x, circle_y, brushWidth):
    radius = brushWidth // 2
    assert len(mask.shape) == 2 or mask.shape[2] == 1, "The channel of mask doesn't fit the opencv format"
    mask_processed = cv2.circle(mask, (circle_x, circle_y), radius, 255)
    return mask_processed


def freeFormMask(img_height, img_width, maxVertex, maxLength, maxBrushWidth, maxAngle):
    mask = np.zeros((img_height, img_width))
    numVertex = random.randint(1, maxVertex)
    startX = random.randint(10, img_width)
    startY = random.randint(10, img_height)
    brushWidth = random.randint(10, maxBrushWidth)
    for i in range(numVertex):
        angle = random.uniform(0, maxAngle)
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = random.randint(10, maxLength)
        mask, endX, endY = draw_line(mask, startX, startY, angle, length, brushWidth)
        startX = startX + int(length * np.sin(angle))
        startY = startY + int(length * np.cos(angle))
        mask = draw_circle(mask, endX, endY, brushWidth)

    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)

    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]

    return mask


if __name__ == "__main__":
    # for stationary mask generation
    # stationary_mask_generator(240, 480, 50, 120)

    # for free-form mask generation
    # mask = freeFormMask(240, 480, 30, 50, 20, np.pi)
    # cv2.imwrite('mask.png', mask)

    # for matrix mask generation
    # img_height, img_width = 240, 480
    # masks = matrix2bbox(240, 480, 20, 20, 5, 4)
    # matrixMask = mbbox2masks(img_height, img_width, masks)
    # cv2.imwrite('matrixMask.png', matrixMask)
    pass


