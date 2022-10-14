import argparse
import glob
from pathlib import Path
import os
import mayavi.mlab as mlab
import numpy as np
import torch
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils.visualize_utils import draw_scenes


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/hz/OpenPCDet/tools/cfgs/kitti_models/3dssd_basa.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default="/home/hz/OpenPCDet/checkpoints/gama_300.pth", help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    dataset, dataloader, sampler = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, workers=8,
                                                batch_size=1, dist=False, training=False, logger=logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    os.environ['CUDA_VISIBLE_DEVICES'] ='4'
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()


    with torch.no_grad():
        for data_dict in dataloader:
            with torch.no_grad():
                load_data_to_gpu(data_dict)
                points = data_dict['points'][:, 1:4]  # (npoints, 3)
                gt_boxes = data_dict['gt_boxes'][0]  # (N_gt, size)
                pred_dicts, ret_dict = model(data_dict)
                pts_samples = torch.load('/home/hz/OpenPCDet/data/kitti/ImageSets/new_xyz.pt')
                # pts_samples = pts_samples.squeeze(0)
                pred_boxes = pred_dicts[0]['pred_boxes']
                recall_dict = {}
                # draw_scenes(points, gt_boxes=gt_boxes, ref_boxes=pred_boxes, ref_scores=None, ref_labels=None)
                #draw_scenes(points, gt_boxes=gt_boxes, ref_boxes=pred_boxes, ref_scores=None, ref_labels=None)
                draw_scenes(
                        points, pts_samples, gt_boxes=gt_boxes,ref_boxes=pred_boxes,
                        ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                    )
                mlab.show(stop=True)
            #break

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
