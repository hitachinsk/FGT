import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA
import argparse
from math import ceil
from glob import glob

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps, ImageFont

from utils.logging_config import logger
from utils.util import make_dirs, bbox_offset


DEFAULT_FPS = 6
MAX_LENGTH = 60


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fps', '--fps',
        type=int, default=DEFAULT_FPS,
        help="Output video FPS"
    )
    parser.add_argument(
        '-v', '--video_dir',
        type=str,
        help="Video directory name"
    )
    parser.add_argument(
        '-vs', '--video_dirs',
        nargs='+',
        type=str,
        help="Video directory names"
    )
    parser.add_argument(
        '-v2', '--video_dir2',
        type=str,
        help="Video directory name"
    )
    parser.add_argument(
        '-sd', '--segms_dir',
        type=str,
        help="Segmentation directory name"
    )
    parser.add_argument(
        '-fgd', '--fg_dir',
        type=str,
        help="Foreground directory name"
    )
    parser.add_argument(
        '-fgfd', '--fg_frames_dir',
        type=str,
        help="Foreground frames directory name"
    )
    parser.add_argument(
        '-fgsd', '--fg_segms_dir',
        type=str,
        help="Foreground segmentations directory name"
    )
    parser.add_argument(
        '-syfd', '--syn_frames_dir',
        type=str,
        help="Synthesized frames directory name"
    )
    parser.add_argument(
        '-bgfd', '--bg_frames_dir',
        type=str,
        help="Background frames directory name"
    )
    parser.add_argument(
        '-rt', '--reader_type',
        type=str,
        help="Type of reader"
    )
    parser.add_argument(
        '-od', '--output_dir',
        type=str,
        help="Output directory name"
    )
    parser.add_argument(
        '-o', '--output_filename',
        type=str, required=True,
        help="Output output filename"
    )
    args = parser.parse_args()
    return args


class Reader:
    def __init__(self, dir_name, read=True, max_length=None, sample_period=1):
        self.dir_name = dir_name
        self.count = 0
        self.max_length = max_length
        self.filenames = []
        self.sample_period = sample_period
        if read:
            if os.path.exists(dir_name):
                # self.filenames = read_filenames_from_dir(dir_name, self.__class__.__name__)
                # ^^^^^ yield None when reading some videos of face forensics data
                # (related to 'Too many levels of symbolic links'?)

                self.filenames = sorted(glob(os.path.join(dir_name, '*')))
                self.filenames = [f for f in self.filenames if os.path.isfile(f)]
                self.filenames = self.filenames[::sample_period][:max_length]
                self.files = self.read_files(self.filenames)
            else:
                self.files = []
                logger.warning(f"Directory {dir_name} not exists!")
        else:
            self.files = []
        self.current_index = 0

    def append(self, file_):
        self.files.append(file_)

    def set_files(self, files):
        self.files = files

    def read_files(self, filenames):
        assert type(filenames) == list, f'filenames is not a list; dirname: {self.dir_name}'
        filenames.sort()
        frames = []
        for filename in filenames:
            file_ = self.read_file(filename)
            frames.append(file_)
        return frames

    def save_files(self, output_dir=None):
        make_dirs(output_dir)
        logger.info(f"Saving {self.__class__.__name__} files to {output_dir}")
        for i, file_ in enumerate(self.files):
            self._save_file(output_dir, i, file_)

    def _save_file(self, output_dir, i, file_):
        raise NotImplementedError("This is an abstract function")

    def read_file(self, filename):
        raise NotImplementedError("This is an abstract function")

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < len(self.files):
            file_ = self.files[self.current_index]
            self.current_index += 1
            return file_
        else:
            self.current_index = 0
            raise StopIteration

    def __getitem__(self, key):
        return self.files[key]

    def __len__(self):
        return len(self.files)


class FrameReader(Reader):
    def __init__(
        self, dir_name, resize=None, read=True, max_length=MAX_LENGTH,
        scale=1, sample_period=1
    ):
        self.resize = resize
        self.scale = scale
        self.sample_period = sample_period
        super().__init__(dir_name, read, max_length, sample_period)

    def read_file(self, filename):
        origin_frame = Image.open(filename)
        size = self.resize if self.resize is not None else origin_frame.size
        origin_frame_resized = origin_frame.resize(
            (int(size[0] * self.scale), int(size[1] * self.scale))
        )
        return origin_frame_resized

    def _save_file(self, output_dir, i, file_):
        if len(self.filenames) == len(self.files):
            name = sorted(self.filenames)[i].split('/')[-1]
        else:
            name = f"frame_{i:04}.png"
        filename = os.path.join(
            output_dir, name
        )
        file_.save(filename, "PNG")

    def write_files_to_video(self, output_filename, fps=DEFAULT_FPS, frame_num_when_repeat_list=[1]):
        logger.info(
            f"Writeing frames to video {output_filename} with FPS={fps}")
        video_writer = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            self.files[0].size
        )
        for frame_num_when_repeat in frame_num_when_repeat_list:
            for frame in self.files:
                frame = frame.convert("RGB")
                frame_cv = np.array(frame)
                frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_RGB2BGR)
                for i in range(frame_num_when_repeat):
                    video_writer.write(frame_cv)
        video_writer.release()


class SynthesizedFrameReader(FrameReader):
    def __init__(
        self, bg_frames_dir, fg_frames_dir,
        fg_segms_dir, segm_bbox_mask_dir, fg_dir, dir_name,
        bboxes_list_dir,
        fg_scale=0.7, fg_location=(48, 27), mask_only=False
    ):
        self.bg_reader = FrameReader(bg_frames_dir)
        self.size = self.bg_reader[0].size
        # TODO: add different location and change scale to var
        self.fg_reader = ForegroundReader(
            fg_frames_dir, fg_segms_dir, fg_dir,
            resize=self.size,
            scale=fg_scale
        )
        self.fg_location = fg_location
        # self.masks = self.fg_reader.masks
        # self.bbox_masks = self.fg_reader.bbox_masks
        super().__init__(dir_name, read=False)
        self.files = self.synthesize_frames(
            self.bg_reader, self.fg_reader, mask_only)
        self.bbox_masks = MaskGenerator(
            segm_bbox_mask_dir, self.size, self.get_bboxeses()
        )
        self.bboxes_list_dir = bboxes_list_dir
        self.bboxes_list = self.get_bboxeses()
        self.save_bboxes()

    def save_bboxes(self):
        make_dirs(self.bboxes_list_dir)
        logger.info(f"Saving bboxes to {self.bboxes_list_dir}")
        for i, bboxes in enumerate(self.bboxes_list):
            save_path = os.path.join(self.bboxes_list_dir, f"bboxes_{i:04}.txt")
            if len(bboxes) > 0:
                np.savetxt(save_path, bboxes[0], fmt='%4u')

    def get_bboxeses(self):
        bboxeses = self.fg_reader.segms.bboxeses
        new_bboxeses = []
        for bboxes in bboxeses:
            new_bboxes = []
            for bbox in bboxes:
                offset_bbox = bbox_offset(bbox, self.fg_location)
                new_bboxes.append(offset_bbox)
            new_bboxeses.append(new_bboxes)
        return new_bboxeses

    def synthesize_frames(self, bg_reader, fg_reader, mask_only=False):
        logger.info(
            f"Synthesizing {bg_reader.dir_name} and {fg_reader.dir_name}"
        )
        synthesized_frames = []
        for i, bg in enumerate(bg_reader):
            if i == len(fg_reader):
                break
            fg = fg_reader[i]
            mask = fg_reader.get_mask(i)
            synthesized_frame = bg.copy()
            if mask_only:
                synthesized_frame.paste(mask, self.fg_location, mask)
            else:
                synthesized_frame.paste(fg, self.fg_location, mask)
            synthesized_frames.append(synthesized_frame)
        return synthesized_frames


class WarpedFrameReader(FrameReader):
    def __init__(self, dir_name, i, ks):
        self.i = i
        self.ks = ks
        super().__init__(dir_name)

    def _save_file(self, output_dir, i, file_):
        filename = os.path.join(
            output_dir,
            f"warped_frame_{self.i:04}_k{self.ks[i]:02}.png"
        )
        file_.save(filename)


class SegmentationReader(FrameReader):
    def __init__(
        self, dir_name,
        resize=None, scale=1
    ):
        super().__init__(
            dir_name, resize=resize, scale=scale
        )

    def read_file(self, filename):
        origin_frame = Image.open(filename)
        mask = ImageOps.invert(origin_frame.convert("L"))
        mask = mask.point(lambda x: 0 if x < 255 else 255, '1')
        size = self.resize if self.resize is not None else origin_frame.size
        mask_resized = mask.resize(
            (int(size[0] * self.scale), int(size[1] * self.scale))
        )
        return mask_resized


class MaskReader(Reader):
    def __init__(self, dir_name, read=True):
        super().__init__(dir_name, read=read)

    def read_file(self, filename):
        mask = Image.open(filename)
        return mask

    def _save_file(self, output_dir, i, file_):
        filename = os.path.join(
            output_dir,
            f"mask_{i:04}.png"
        )
        file_.save(filename)

    def get_bboxes(self, i):
        # TODO: save bbox instead of looking for one
        mask = self.files[i]
        mask = ImageOps.invert(mask.convert("L")).convert("1")
        mask = np.array(mask)
        image, contours, hier = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bboxes = []
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            bbox = ((x, y), (x + w - 1, y + h - 1))
            bboxes.append(bbox)
        return bboxes

    def get_bbox(self, i):
        # TODO: save bbox instead of looking for one
        mask = self.files[i]
        mask = ImageOps.invert(mask.convert("L"))
        mask = np.array(mask)
        image, contours, hier = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            bbox = ((x, y), (x + w - 1, y + h - 1))
            return bbox


class MaskGenerator(Reader):
    def __init__(
        self, mask_output_dir, size, bboxeses, save_masks=True
    ):
        self.bboxeses = bboxeses
        self.size = size
        super().__init__(mask_output_dir, read=False)
        self.files = self.generate_masks()
        if save_masks:
            make_dirs(mask_output_dir)
            self.save_files(mask_output_dir)

    def _save_file(self, output_dir, i, file_):
        filename = os.path.join(
            output_dir,
            f"mask_{i:04}.png"
        )
        file_.save(filename)

    def get_bboxes(self, i):
        return self.bboxeses[i]

    def generate_masks(self):
        masks = []
        for i in range(len(self.bboxeses)):
            mask = self.generate_mask(i)
            masks.append(mask)
        return masks

    def generate_mask(self, i):
        bboxes = self.bboxeses[i]
        mask = Image.new("1", self.size, 1)
        draw = ImageDraw.Draw(mask)
        for bbox in bboxes:
            draw.rectangle(
                bbox, fill=0
            )
        return mask


class ForegroundReader(FrameReader):
    def __init__(
        self, frames_dir, segms_dir, dir_name,
        resize=None, scale=1
    ):
        self.frames_dir = frames_dir
        self.segms_dir = segms_dir
        self.frames = FrameReader(
            frames_dir,
            resize=resize, scale=scale
        )
        self.segms = SegmentationReader(
            segms_dir, resize=resize, scale=scale
        )
        super().__init__(dir_name, read=False)
        self.masks = self.segms.masks
        # self.bbox_masks = self.segms.bbox_masks
        self.files = self.generate_fg_frames(self.frames, self.segms)

    def get_mask(self, i):
        return self.masks[i]

    def generate_fg_frames(self, frames, segms):
        logger.info(
            f"Generating fg frames from {self.frames_dir} and {self.segms_dir}"
        )
        fg_frames = []
        for i, frame in enumerate(frames):
            mask = self.masks[i]
            fg_frame = Image.new("RGB", frame.size, (0, 0, 0))
            fg_frame.paste(
                frame, (0, 0),
                mask
            )
            fg_frames.append(fg_frame)
        return fg_frames


class CompareFramesReader(FrameReader):
    def __init__(self, dir_names, col=2, names=[], mask_dir=None):
        self.videos = []
        for dir_name in dir_names:
            # If a method fails on this video, use None to indicate the situation
            try:
                self.videos.append(FrameReader(dir_name))
            except AssertionError:
                self.videos.append(None)
        if mask_dir is not None:
            self.masks = MaskReader(mask_dir)
        self.names = names
        self.files = self.combine_videos(self.videos, col)

    def combine_videos(self, videos, col=2, edge_offset=35, h_start_offset=35):
        combined_frames = []
        w, h = videos[0][0].size
        # Prevent the first method fails and have a "None" as its video
        i = 0
        while videos[i] is None:
            i += 1
        length = len(videos[i])
        video_num = len(videos)
        row = ceil(video_num / col)
        for frame_idx in range(length):
            width = col * w + (col - 1) * edge_offset
            height = row * h + (row - 1) * edge_offset + h_start_offset
            combined_frame = Image.new("RGBA", (width, height))
            draw = ImageDraw.Draw(combined_frame)
            for i, video in enumerate(videos):
                # Give the failed method a black output
                if video is None or frame_idx >= len(video):
                    failed = True
                    frame = Image.new("RGBA", (w, h))
                else:
                    frame = video[frame_idx].convert("RGBA")
                    failed = False

                f_x = (i % col) * (w + edge_offset)
                f_y = (i // col) * (h + edge_offset) + h_start_offset
                combined_frame.paste(frame, (f_x, f_y))

                # Draw name
                font = ImageFont.truetype("DejaVuSans.ttf", 12)
                # font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
                # font = ImageFont.truetype("timesbd.ttf", 14)
                name = self.names[i] if not failed else f'{self.names[i]} (failed)'
                draw.text(
                    (f_x + 10, f_y - 20),
                    name, (255, 255, 255), font=font
                )

            combined_frames.append(combined_frame)
        return combined_frames


class BoundingBoxesListReader(Reader):
    def __init__(
        self, dir_name, resize=None, read=True, max_length=MAX_LENGTH,
        scale=1
    ):
        self.resize = resize
        self.scale = scale
        super().__init__(dir_name, read, max_length)

    def read_file(self, filename):
        bboxes = np.loadtxt(filename, dtype=int)
        bboxes = [bboxes.tolist()]
        return bboxes


def save_frames_to_dir(frames, dirname):
    reader = FrameReader(dirname, read=False)
    reader.set_files(frames)
    reader.save_files(dirname)


if __name__ == "__main__":
    args = parse_args()
    if args.reader_type is None:
        reader = FrameReader(args.video_dir)
    elif args.reader_type == 'fg':
        reader = ForegroundReader(
            args.video_dir, args.segms_dir, args.fg_dir)
    elif args.reader_type == 'sy':
        reader = SynthesizedFrameReader(
            args.bg_frames_dir, args.fg_frames_dir,
            args.fg_segms_dir, args.fg_dir, args.syn_frames_dir
        )
    elif args.reader_type == 'com':
        reader = CompareFramesReader(
            args.video_dirs
        )
    reader.write_files_to_video(
        os.path.join(args.output_dir, args.output_filename),
        fps=args.fps
    )
