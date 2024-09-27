# -*- coding: utf-8 -*-
from typing import List
import os
from PIL import Image
import numpy as np
import cv2
from shapely.geometry import Polygon
from collections import OrderedDict


def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "y", "true", "t", "1")


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def draw_boxes(image: Image.Image, boxes: List[List[List[int]]], line_thickness: int = 2) -> Image.Image:
    img = np.array(image)
    for i, box in enumerate(boxes):
        poly_ = np.array(box_to_poly(box)).astype(np.int32).reshape((-1))
        poly_ = poly_.reshape(-1, 2)
        cv2.polylines(img, [poly_.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=line_thickness)
        ptColor = (0, 255, 255)
    return Image.fromarray(img)


def draw_polygons(image: Image.Image, polygons: List[List[List[int]]], line_thickness: int = 2) -> Image.Image:
    img = np.array(image)
    for i, poly in enumerate(polygons):
        poly_ = np.array(poly).astype(np.int32).reshape((-1))
        poly_ = poly_.reshape(-1, 2)
        cv2.polylines(img, [poly_.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=line_thickness)
        ptColor = (0, 255, 255)
    return Image.fromarray(img)


def box_to_poly(box: List[List[int]]) -> List[List[int]]:
    return [box[0], [box[0][0], box[1][1]], box[1], [box[1][0], box[0][1]]]


def boxes_area(bboxes: List[List[List[int]]]) -> int:
    total_S = 0
    for box in bboxes:
        pgon = Polygon(box_to_poly(box)) 
        S = pgon.area
        total_S+=S
    return total_S


def polygons_area(polygons: List[List[List[int]]]) -> int:
    total_S = 0
    for poly in polygons:
        pgon = Polygon(poly) 
        S = pgon.area
        total_S+=S
    return total_S
