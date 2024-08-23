import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp #混合精度训练
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# import val as validate
# from models.experimental import attempt_load
#TODO 模型部分搭建，vgg19 1）自己搭设模型 2）用torch.hub 加载预训练模型 3）改模型输出，
from models.yolo import ClassificationModel
# TODO 数据加载部分，需要重新编写
from utils.dataloaders import create_classification_dataloader
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    WorkingDirectory,
    colorstr,
    download,
    increment_path,
    init_seeds,
    print_args,
    yaml_save,
)
from utils.loggers import GenericLogger
from utils.plots import imshow_cls
from utils.torch_utils import (
    ModelEMA,
    model_info,
    reshape_classifier_output,
    select_device,
    smart_optimizer,
    smartCrossEntropyLoss,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))