#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import paddle
import paddle.vision as paddlevision

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def nms(boxes, scores, thresh):
    """Apply classic DPM-style greedy NMS."""
    boxes = boxes.numpy()
    scores = scores.numpy()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = boxes.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    return keep


def batched_nms(
    boxes: paddle.Tensor,
    scores: paddle.Tensor,
    idxs: paddle.Tensor,
    iou_threshold: float,
) -> paddle.Tensor:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return paddle.empty((0,), dtype=paddle.int64)
    max_coordinate = boxes.max()
    offsets = paddle.cast(idxs, boxes.dtype) * (max_coordinate + paddle.to_tensor(1, dtype=boxes.dtype))
    boxes_for_nms = boxes + offsets.unsqueeze(-1)
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return paddle.to_tensor(keep)


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = paddle.empty(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
        # Get score and class with highest confidence
        class_conf = paddle.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        class_pred = paddle.argmax(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = paddle.concat((image_pred[:, :5], class_conf, paddle.cast(class_pred, class_conf.dtype)), 1)
        idx = paddle.masked_select(paddle.arange(0, detections.shape[0]), conf_mask)
        detections = detections[idx]
        if not detections.shape[0] or detections.ndim == 1:
            continue

        nms_out_index = batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = paddle.concat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = paddle.max(bboxes_a[:, :2], bboxes_b[:, :2])
        br = paddle.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
        area_a = paddle.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = paddle.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = paddle.maximum(
            (bboxes_a[:, :2] - bboxes_a[:, 2:] / 2).unsqueeze(1),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = paddle.minimum(
            (bboxes_a[:, :2] + bboxes_a[:, 2:] / 2).unsqueeze(1),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = paddle.prod(bboxes_a[:, 2:], 1)
        area_b = paddle.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).cast(tl.dtype).prod(axis=2)
    area_i = paddle.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a.unsqueeze(-1) + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
