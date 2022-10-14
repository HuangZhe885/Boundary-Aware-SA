import torch
import numpy as np
import yaml
from easydict import EasyDict
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu, model_fn_decorator
from pcdet.datasets import build_dataloader
import random
from pcdet.utils import common_utils
import time
import open3d
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  #for accelerating the running

setup_seed(1024)

def parse_args():
    parser = argparse.ArgumentParser(description='OpenPCDet benchmark a model')
    parser.add_argument('--config', default="/home/hz/OpenPCDet/tools/cfgs/kitti_models/3dssd_basa.yaml",
                        help='test config file path')
    parser.add_argument('--checkpoint', default="/home/hz/OpenPCDet/checkpoints/gama_300.pth", help='checkpoint file')
    parser.add_argument('--samples', default=2000, type=int, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    dataset, dataloader, sampler = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                                    batch_size=1, dist=False, training=False, logger=logger)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset).cuda()

    # model.load_params_from_file(filename=args.checkpoint, logger=logger)
    model.eval()

    # the first several iterations may be very slow so skip them
    # #有个地方说，扔掉前5张图的耗时【模型加载啥的初始化工作不算进去】
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    with torch.no_grad():
        for i, data_dict in enumerate(dataloader):
            load_data_to_gpu(data_dict)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            pred_dicts, ret_dict = model(data_dict)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done image [{i + 1:<3}/ {args.samples}], '
                          f'fps: {fps:.1f} img / s')

            if (i + 1) == args.samples:
                pure_inf_time += elapsed
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.1f} img / s')
                break

if __name__ == '__main__':
    main()
