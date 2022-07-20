from .STTN_mask import create_random_shape_with_random_motion

import logging
logger = logging.getLogger('base')


def initialize_mask(videoLength, dataInfo):
    from .MaskModel import RandomMask
    from .MaskModel import MidRandomMask
    from .MaskModel import MatrixMask
    from .MaskModel import FreeFormMask
    from .MaskModel import StationaryMask

    return {'random': RandomMask(videoLength, dataInfo),
            'mid': MidRandomMask(videoLength, dataInfo),
            'matrix': MatrixMask(videoLength, dataInfo),
            'free': FreeFormMask(videoLength, dataInfo),
            'stationary': StationaryMask(videoLength, dataInfo)
            }


def create_mask(maskClass, form):
    if form == 'mix':
        from random import randint
        candidates = list(maskClass.keys())
        candidate_index = randint(0, len(candidates) - 1)
        return maskClass[candidates[candidate_index]]()
    return maskClass[form]()