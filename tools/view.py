import cv2
import torch
import numpy as np
import yaml
from easydict import EasyDict
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu, model_fn_decorator
from pcdet.datasets import build_dataloader
import random
from pcdet.utils import common_utils, box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import time

import mayavi.mlab as mlab
from visual_utils.visualize_utils import draw_scenes


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.cuda.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.benchmark = False
#     # torch.backends.cudnn.benchmark = True #for accelerating the running

# setup_seed(666)

cfg_file = "/home/hz/OpenPCDet/tools/cfgs/kitti_models/3dssd_basa.yaml"
cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger()
dataset, dataloader, sampler = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, workers=8,
                                                batch_size=4, dist=False, training=False, logger=logger)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset).cuda()
# for name, param in model.named_parameters():
#     print(name, param.sum(), param.size())

model.load_params_from_file(filename="/home/hz/OpenPCDet/checkpoints/basa_car_kitti_80.pth",
                             logger=logger)


model.eval()


for data_dict in dataloader:
    with torch.no_grad():
        load_data_to_gpu(data_dict)
        points = data_dict['points'][:, 1:4]  # (npoints, 3)
        gt_boxes = data_dict['gt_boxes'][0]  # (N_gt, size)
        pred_dicts, ret_dict = model(data_dict)
        pred_boxes = pred_dicts[0]['pred_boxes']
        recall_dict = {}
        # draw_scenes(points, gt_boxes=gt_boxes, ref_boxes=pred_boxes, ref_scores=None, ref_labels=None)
        #draw_scenes(points, gt_boxes=gt_boxes, ref_boxes=pred_boxes, ref_scores=None, ref_labels=None)
        draw_scenes(
                points, gt_boxes=gt_boxes,ref_boxes=pred_boxes,
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
        mlab.show(stop=True)
    break