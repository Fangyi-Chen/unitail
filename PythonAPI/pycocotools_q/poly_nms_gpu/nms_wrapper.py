# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .poly_nms import poly_gpu_nms
from .poly_overlaps import poly_overlaps
import numpy as np 


def poly_nms_gpu(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    dets = dets.astype(np.float32)
    return poly_gpu_nms(dets, thresh, device_id=0)

def poly_overlaps_gpu(bboxes1, bboxes2):
    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return np.empty((0), dtype=np.float32)
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    return poly_overlaps(bboxes1, bboxes2, device_id=0)

