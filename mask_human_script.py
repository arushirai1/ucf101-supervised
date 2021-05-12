# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import torch
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from google.colab.patches import cv2_imshow
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
im = cv2.imread("./input.jpg")

def mask_frame(predictor, img):
  outputs = predictor(img)
  if 0 in outputs["instances"].pred_classes:
    pred_classes, pred_boxes = zip(*[(c, box) for c, box in zip(outputs["instances"].pred_classes, outputs["instances"].pred_boxes) if c == 0])
    avg_color=np.mean(np.mean(img, axis=0), axis=0)
    for mask_coord in pred_boxes:
      mask_coord = mask_coord.type(torch.IntTensor).tolist()
      img=cv2.rectangle(img, tuple(mask_coord[0:2]), tuple(mask_coord[2:]), avg_color, -1)
  return img

# save as png
image = mask_frame(predictor, im)
cv2.imwrite("title.png", image)