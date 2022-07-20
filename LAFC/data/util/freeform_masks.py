import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # NOQA

import numpy as np
import argparse
from PIL import Image

from .mask_generators import get_video_masks_by_moving_random_stroke, get_masked_ratio
from .util import make_dirs, make_dir_under_root, get_everything_under
from .readers import MaskReader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-od', '--output_dir',
        type=str,
        help="Output directory name"
    )
    parser.add_argument(
        '-im',
        '--image_masks', action='store_true',
        help="Set this if you want to generate independent masks in one directory."
    )
    parser.add_argument(
        '-vl', '--video_len',
        type=int,
        help="Maximum video length (i.e. #mask)"
    )
    parser.add_argument(
        '-ns', '--num_stroke',
        type=int,
        help="Number of stroke in one mask"
    )
    parser.add_argument(
        '-nsb', '--num_stroke_bound',
        type=int,
        nargs=2,
        help="Upper/lower bound of number of stroke in one mask"
    )
    parser.add_argument(
        '-n',
        type=int,
        help="Number of mask to generate"
    )
    parser.add_argument(
        '-sp',
        '--stroke_preset',
        type=str,
        default='rand_curve',
        help="Preset of the stroke parameters"
    )
    parser.add_argument(
        '-iw',
        '--image_width',
        type=int,
        default=320
    )
    parser.add_argument(
        '-ih',
        '--image_height',
        type=int,
        default=180
    )
    parser.add_argument(
        '--cluster_by_area',
        action='store_true'
    )
    parser.add_argument(
        '--leave_boarder_unmasked',
        type=int,
        help='Set this to a number, then a copy of the mask where the mask of boarder is erased.'
    )
    parser.add_argument(
        '--redo_without_generation',
        action='store_true',
        help='Set this, and the script will skip the generation and redo the left tasks'
             '(uncluster -> erase boarder -> re-cluster)'
    )
    args = parser.parse_args()
    return args


def get_stroke_preset(stroke_preset):
    if stroke_preset == 'object_like':
        return {
            "nVertexBound": [5, 30],
            "maxHeadSpeed": 15,
            "maxHeadAcceleration": (10, 1.5),
            "brushWidthBound": (20, 50),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 10,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": None,
            "maxInitSpeed": 10,
        }
    elif stroke_preset == 'object_like_middle':
        return {
            "nVertexBound": [5, 15],
            "maxHeadSpeed": 8,
            "maxHeadAcceleration": (4, 1.5),
            "brushWidthBound": (20, 50),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 5,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": None,
            "maxInitSpeed": 10,
        }
    elif stroke_preset == 'object_like_small':
        return {
            "nVertexBound": [5, 20],
            "maxHeadSpeed": 7,
            "maxHeadAcceleration": (3.5, 1.5),
            "brushWidthBound": (10, 30),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 5,
            "maxLineAcceleration": (3, 0.5),
            "boarderGap": None,
            "maxInitSpeed": 4,
        }
    elif stroke_preset == 'rand_curve':
        return {
            "nVertexBound": [10, 30],
            "maxHeadSpeed": 20,
            "maxHeadAcceleration": (15, 0.5),
            "brushWidthBound": (3, 10),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 3,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": None,
            "maxInitSpeed": 6
        }
    elif stroke_preset == 'rand_curve_small':
        return {
            "nVertexBound": [6, 22],
            "maxHeadSpeed": 12,
            "maxHeadAcceleration": (8, 0.5),
            "brushWidthBound": (2.5, 5),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 1.5,
            "maxLineAcceleration": (3, 0.5),
            "boarderGap": None,
            "maxInitSpeed": 3
        }
    else:
        raise NotImplementedError(f'The stroke presetting "{stroke_preset}" does not exist.')


def copy_masks_without_boarder(root_dir, args):
    def erase_mask_boarder(mask, gap):
        pix = np.asarray(mask).astype('uint8') * 255
        pix[:gap, :] = 255
        pix[-gap:, :] = 255
        pix[:, :gap] = 255
        pix[:, -gap:] = 255
        return Image.fromarray(pix).convert('1')

    wo_boarder_dir = root_dir + '_noBoarder'
    shutil.copytree(root_dir, wo_boarder_dir)

    for i, filename in enumerate(get_everything_under(wo_boarder_dir)):
        if args.image_masks:
            mask = Image.open(filename)
            mask_wo_boarder = erase_mask_boarder(mask, args.leave_boarder_unmasked)
            mask_wo_boarder.save(filename)
        else:
            # filename is a diretory containing multiple mask files
            for f in get_everything_under(filename, pattern='*.png'):
                mask = Image.open(f)
                mask_wo_boarder = erase_mask_boarder(mask, args.leave_boarder_unmasked)
                mask_wo_boarder.save(f)

    return wo_boarder_dir


def cluster_by_masked_area(root_dir, args):
    clustered_dir = root_dir + '_clustered'
    make_dirs(clustered_dir)
    radius = 5

    # all masks with ratio in x +- radius will be stored in sub-directory x
    clustered_centors = np.arange(radius, 100, radius * 2)
    clustered_subdirs = []
    for c in clustered_centors:
        # make sub-directories for each ratio range
        clustered_subdirs.append(make_dir_under_root(clustered_dir, str(c)))

    for i, filename in enumerate(get_everything_under(root_dir)):
        if args.image_masks:
            ratio = get_masked_ratio(Image.open(filename))
        else:
            # filename is a diretory containing multiple mask files
            ratio = np.mean([
                get_masked_ratio(Image.open(f))
                for f in get_everything_under(filename, pattern='*.png')
            ])

        # find the nearest centor
        for i, c in enumerate(clustered_centors):
            if c - radius <= ratio * 100 <= c + radius:
                shutil.move(filename, clustered_subdirs[i])
                break

    shutil.rmtree(root_dir)
    os.rename(clustered_dir, root_dir)


def decide_nStroke(args):
    if args.num_stroke is not None:
        return args.num_stroke
    elif args.num_stroke_bound is not None:
        return np.random.randint(args.num_stroke_bound[0], args.num_stroke_bound[1])
    else:
        raise ValueError('One of "-ns" or "-nsb" is needed')


def main(args):
    preset = get_stroke_preset(args.stroke_preset)
    make_dirs(args.output_dir)

    if args.redo_without_generation:
        assert(len(get_everything_under(args.output_dir)) > 0)
        # put back clustered masks
        for clustered_subdir in get_everything_under(args.output_dir):
            if not os.path.isdir(clustered_subdir):
                continue
            for f in get_everything_under(clustered_subdir):
                shutil.move(f, args.output_dir)
            os.rmdir(clustered_subdir)

    else:
        if args.image_masks:
            for i in range(args.n):
                nStroke = decide_nStroke(args)
                mask = get_video_masks_by_moving_random_stroke(
                    video_len=1, imageWidth=args.image_width, imageHeight=args.image_height,
                    nStroke=nStroke, **preset
                )[0]
                mask.save(os.path.join(args.output_dir, f'{i:07d}.png'))

        else:
            for i in range(args.n):
                mask_dir = make_dir_under_root(args.output_dir, f'{i:05d}')
                mask_reader = MaskReader(mask_dir, read=False)

                nStroke = decide_nStroke(args)
                masks = get_video_masks_by_moving_random_stroke(
                    imageWidth=args.image_width, imageHeight=args.image_height,
                    video_len=args.video_len, nStroke=nStroke, **preset)

                mask_reader.set_files(masks)
                mask_reader.save_files(output_dir=mask_reader.dir_name)

    if args.leave_boarder_unmasked is not None:
        dir_leave_boarder = copy_masks_without_boarder(args.output_dir, args)
        if args.cluster_by_area:
            cluster_by_masked_area(dir_leave_boarder, args)

    if args.cluster_by_area:
        cluster_by_masked_area(args.output_dir, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
