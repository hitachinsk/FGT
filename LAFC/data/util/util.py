import os
import argparse
import shutil
from glob import glob

import numpy as np
from PIL import Image

from utils.logging_config import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--video_dir',
        type=str,
        help="Video directory name"
    )
    parser.add_argument(
        '-fl', '--flow_dir',
        type=str,
        help="Optical flow ground truth directory name"
    )
    parser.add_argument(
        '-od', '--output_dir',
        type=str,
        help="Output directory name"
    )
    parser.add_argument(
        '-o', '--output_filename',
        type=str,
        help="Output output filename"
    )
    args = parser.parse_args()
    return args


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        logger.info(f"Directory {dir_name} made")


ensure_dir = make_dirs


def make_dir_under_root(root_dir, name):
    full_dir_name = os.path.join(root_dir, name)
    make_dirs(full_dir_name)
    return full_dir_name


def rm_dirs(dir_name, ignore_errors=False):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors)
        logger.info(f"Directory {dir_name} removed")


def read_dirnames_under_root(root_dir, skip_list=[]):
    dirnames = [
        name for i, name in enumerate(sorted(os.listdir(root_dir)))
        if (os.path.isdir(os.path.join(root_dir, name))
            and name not in skip_list
            and i not in skip_list)
    ]
    logger.info(f"Reading directories under {root_dir}, exclude {skip_list}, num: {len(dirnames)}")
    return dirnames


def bbox_offset(bbox, location):
    x0, y0 = location
    (x1, y1), (x2, y2) = bbox
    return ((x1 + x0, y1 + y0), (x2 + x0, y2 + y0))


def cover2_bbox(bbox1, bbox2):
    x1 = min(bbox1[0][0], bbox2[0][0])
    y1 = min(bbox1[0][1], bbox2[0][1])
    x2 = max(bbox1[1][0], bbox2[1][0])
    y2 = max(bbox1[1][1], bbox2[1][1])
    return ((x1, y1), (x2, y2))


def extend_r_bbox(bbox, w, h, r):
    (x1, y1), (x2, y2) = bbox
    x1 = max(x1 - r, 0)
    x2 = min(x2 + r, w)
    y1 = max(y1 - r, 0)
    y2 = min(y2 + r, h)
    return ((x1, y1), (x2, y2))


def mean_squared_error(A, B):
    return np.square(np.subtract(A, B)).mean()


def bboxes_to_mask(size, bboxes):
    mask = Image.new("L", size, 255)
    mask = np.array(mask)
    for bbox in bboxes:
        try:
            (x1, y1), (x2, y2) = bbox
        except Exception:
            (x1, y1, x2, y2) = bbox

        mask[y1:y2, x1:x2] = 0
    mask = Image.fromarray(mask.astype("uint8"))
    return mask


def get_extended_from_box(img_size, box, patch_size):
    def _decide_patch_num(box_width, patch_size):
        num = np.ceil(box_width / patch_size).astype(np.int)
        if (num * patch_size - box_width) < (patch_size // 2):
            num += 1
        return num

    x1, y1 = box[0]
    x2, y2 = box[1]
    new_box = (x1, y1, x2 - x1, y2 - y1)
    box_x_start, box_y_start, box_x_size, box_y_size = new_box

    patchN_x = _decide_patch_num(box_x_size, patch_size)
    patchN_y = _decide_patch_num(box_y_size, patch_size)

    extend_x = (patch_size * patchN_x - box_x_size) // 2
    extend_y = (patch_size * patchN_y - box_y_size) // 2
    img_x_size = img_size[0]
    img_y_size = img_size[1]

    x_start = max(0, box_x_start - extend_x)
    x_end = min(box_x_start - extend_x + patchN_x * patch_size, img_x_size)

    y_start = max(0, box_y_start - extend_y)
    y_end = min(box_y_start - extend_y + patchN_y * patch_size, img_y_size)
    x_start, y_start, x_end, y_end = int(x_start), int(y_start), int(x_end), int(y_end)
    extented_box = ((x_start, y_start), (x_end, y_end))
    return extented_box


# code modified from https://github.com/WonwoongCho/Generative-Inpainting-pytorch/blob/master/util.py
def spatial_discounting_mask(mask_width, mask_height, discounting_gamma):
    """Generate spatial discounting mask constant.
    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Returns:
        np.array: spatial discounting mask
    """
    gamma = discounting_gamma
    mask_values = np.ones((mask_width, mask_height), dtype=np.float32)
    for i in range(mask_width):
        for j in range(mask_height):
            mask_values[i, j] = max(
                gamma**min(i, mask_width - i),
                gamma**min(j, mask_height - j))

    return mask_values


def bboxes_to_discounting_loss_mask(img_size, bboxes, discounting_gamma=0.99):
    mask = np.zeros(img_size, dtype=np.float32) + 0.5
    for bbox in bboxes:
        try:
            (x1, y1), (x2, y2) = bbox
        except Exception:
            (x1, y1, x2, y2) = bbox
        mask_width, mask_height = y2 - y1, x2 - x1
        mask[y1:y2, x1:x2] = spatial_discounting_mask(mask_width, mask_height, discounting_gamma)
    return mask


def find_proper_window(image_size, bbox_point):
    '''
        parameters:
            image_size(2-tuple): (height, width)
            bbox_point(2-2-tuple): (first_point, last_point)
        return values:
            window left-up point, (2-tuple)
            window right-bottom point, (2-tuple)
    '''
    bbox_height = bbox_point[1][0] - bbox_point[0][0]
    bbox_width = bbox_point[1][1] - bbox_point[0][1]

    window_size = min(
        max(bbox_height, bbox_width) * 2,
        image_size[0], image_size[1]
    )
    # Limit min window size due to the requirement of VGG16
    window_size = max(window_size, 32)

    horizontal_span = window_size - (bbox_point[1][1] - bbox_point[0][1])
    vertical_span = window_size - (bbox_point[1][0] - bbox_point[0][0])

    top_bound, bottom_bound = bbox_point[0][0] - \
        vertical_span // 2, bbox_point[1][0] + vertical_span // 2
    left_bound, right_bound = bbox_point[0][1] - \
        horizontal_span // 2, bbox_point[1][1] + horizontal_span // 2

    if left_bound < 0:
        right_bound += 0 - left_bound
        left_bound += 0 - left_bound
    elif right_bound > image_size[1]:
        left_bound -= right_bound - image_size[1]
        right_bound -= right_bound - image_size[1]

    if top_bound < 0:
        bottom_bound += 0 - top_bound
        top_bound += 0 - top_bound
    elif bottom_bound > image_size[0]:
        top_bound -= bottom_bound - image_size[0]
        bottom_bound -= bottom_bound - image_size[0]

    return (top_bound, left_bound), (bottom_bound, right_bound)


def drawrect(drawcontext, xy, outline=None, width=0, partial=None):
    (x1, y1), (x2, y2) = xy
    if partial is None:
        points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
        drawcontext.line(points, fill=outline, width=width)
    else:
        drawcontext.line([(x1, y1), (x1, y1 + partial)], fill=outline, width=width)
        drawcontext.line([(x1 + partial, y1), (x1, y1)], fill=outline, width=width)

        drawcontext.line([(x2, y1), (x2, y1 + partial)], fill=outline, width=width)
        drawcontext.line([(x2, y1), (x2 - partial, y1)], fill=outline, width=width)

        drawcontext.line([(x1, y2), (x1 + partial, y2)], fill=outline, width=width)
        drawcontext.line([(x1, y2), (x1, y2 - partial)], fill=outline, width=width)

        drawcontext.line([(x2 - partial, y2), (x2, y2)], fill=outline, width=width)
        drawcontext.line([(x2, y2), (x2, y2 - partial)], fill=outline, width=width)


def get_everything_under(root_dir, pattern='*', only_dirs=False, only_files=False):
    assert not(only_dirs and only_files), 'You will get nothnig '\
        'when "only_dirs" and "only_files" are both set to True'
    everything = sorted(glob(os.path.join(root_dir, pattern)))
    if only_dirs:
        everything = [f for f in everything if os.path.isdir(f)]
    if only_files:
        everything = [f for f in everything if os.path.isfile(f)]

    return everything


def read_filenames_from_dir(dir_name, reader, max_length=None):
    logger.debug(
        f"{reader} reading files from {dir_name}")
    filenames = []
    for root, dirs, files in os.walk(dir_name):
        assert len(dirs) == 0, f"There are direcories: {dirs} in {root}"
        assert len(files) != 0, f"There are no files in {root}"
        filenames = [os.path.join(root, name) for name in sorted(files)]
        for name in filenames:
            logger.debug(name)
        if max_length is not None:
            return filenames[:max_length]
        return filenames
