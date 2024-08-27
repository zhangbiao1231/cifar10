import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from streamlit.watcher import watch_dir
from torch.cuda import amp #混合精度训练
from tqdm import tqdm
from urllib3.filepost import writer

from models.common import Classify
from utils.downloads import curl_download, attempt_download
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate
from models.experimental import attempt_load
from models.classify import (
    Model,
    ClassificationModel,
)
from utils.dataloaders import create_classification_dataloader
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    WorkingDirectory,
    colorstr,
    intersect_dicts,
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
    smart_resume,
    smartCrossEntropyLoss,
    torch_distributed_zero_first,
    de_parallel,
    EarlyStopping,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

def train(opt,device):
    """Trains a YOLOv5 model, managing datasets, model optimization, logging, and saving checkpoints."""
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, weights, data, bs, cfg, epochs, resume, nw, imgsz, pretrained = (
        opt.save_dir,
        opt.weights,
        Path(opt.data),
        opt.batch_size,
        opt.cfg,
        opt.epochs,
        opt.resume,
        min(os.cpu_count() - 1, opt.workers),
        opt.imgsz,
        str(opt.pretrained).lower() == "true",
    )
    cuda = device.type != "cpu"

    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True,exist_ok=True) # make dir
    last, best = wdir / "last.pt", wdir / "best.pt"

    # Save run settings
    yaml_save(save_dir / "opt.yaml" ,vars(opt))

    # Logger
    logger = GenericLogger(opt=opt,console_logger=LOGGER) if RANK in {-1,0} else None

    # Download Dataset
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        data_dir  = data if data.is_dir() else (DATASETS_DIR / data)
        if not data_dir.is_dir():
            LOGGER.info(f"\nDataset not found ⚠️, missing path {data_dir}, attempting download...")
            t = time.time()
            if str(data) == "cifar10":
                subprocess.run(["bash",str(ROOT / "data/scripts/get_cifar10.sh")],shell=True,check=True)
            else:
                url = ""
                download(url,dir=data_dir.parent)
            s= f"Dataset download success ✅ ({time.time() - t:.1f}s) ,saved to {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

    # DataLoaders
    nc = len([x for x in (data_dir / "train_valid_test/train").glob("*") if x.is_dir()]) # number of classes
    train_dir = data_dir / "train_valid_test/train" if (data_dir / "train_valid_test/train").exists() else data_dir / "train_valid_test/train_valid"
    trainloader = create_classification_dataloader(
        path=train_dir,
        imgsz=imgsz,
        batch_size=bs // WORLD_SIZE,
        augment=True,
        cache=opt.cache,
        rank=LOCAL_RANK,
        workers=nw,
        shuffle=True,
    )
    valid_dir = data_dir / "train_valid_test/valid" if (data_dir / "train_valid_test/valid").exists() else  data_dir / "train_valid_test/test"
    if RANK in {-1,0}:
        validloader = create_classification_dataloader(
            path=valid_dir,
            imgsz=imgsz,
            batch_size=bs // WORLD_SIZE * 2,
            augment=False,
            cache=opt.cache,
            rank=LOCAL_RANK,
            workers=nw,
            shuffle=False,
    )
    # Model
    pretrained = str(weights).endswith(".pt")
    if pretrained:
        # ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        # model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc).to(device)  # create
        # model = ClassificationModel(cfg=None, model=model, nc=nc, cutoff=9) #TODO 这里模型经过裁剪，与.yaml生成的不同
        # exclude = []  # exclude keys
        # csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # model.load_state_dict(csd, strict=False)  # load
        # LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
        model = nn.ModuleList()
        file = Path(str(weights).strip().replace("'", ""))
        ckpt = torch.load(file, map_location="cpu")  # load
        csd = ckpt["model"].to(device).float()  # FP32 model
        model.append(csd.eval())
        model = model[-1]
    else:
        with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
            if Path(opt.model).is_file() or opt.model.endswith(".pt"):
                model = attempt_load(opt.model, device=device, fuse=False)
            elif opt.model in torchvision.models.__dict__:# TorchVision models i.e. resnet50, efficientnet_b0
                model = torchvision.models.__dict__[opt.model](weights=None)
                reshape_classifier_output(model, nc)  # update class count
            else:
                model = Model(cfg= cfg, ch=3, nc=nc).to(device)  # create
            model = ClassificationModel(cfg = None, model = model, nc = nc, cutoff = 9)
            # print(model)

    for p in model.parameters():
        p.requires_grad = True  # for training
    model = model.to(device)

    # Info
    if RANK in {-1, 0}:
        model.names = trainloader.dataset.classes  # attach class names
        model.transforms = validloader.dataset.torch_transforms  # attach inference transforms
        model_info(model)
        if opt.verbose:
            LOGGER.info(model)
        images, labels = next(iter(trainloader))
        file = imshow_cls(images[:16], labels[:16], names=model.names, f=save_dir / "train_images.jpg")
        logger.log_images(file, name="Train Examples")
        logger.log_graph(model, imgsz)  # log model

        # Optimizer
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)

    # Scheduler
    lrf = 0.01  # final lr (fraction of lr0)

    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    def lf(x):
        """Linear learning rate scheduler function, scaling learning rate from initial value to `lrf` over `epochs`."""
        return (1 - x / epochs) * (1 - lrf) + lrf  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch , final_epoch = 0.0, 0, None # initialize
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # Train
    t0 = time.time()
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    stopper, stop = EarlyStopping(patience=opt.patience,min_delta=opt.min_delta), False
    val = valid_dir.stem.split('/')[-1]  # 'valid' or 'test'
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} valid\n'
        f'Using {nw * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting {opt.model} training on {data} dataset with {nc} classes for {epochs} epochs...\n\n'
        f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}"
    )
    for epoch in range(start_epoch,epochs):  # loop over the dataset multiple times# epoch -----------------------------
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        model.train()
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format=TQDM_BAR_FORMAT)
        for i, (images, labels) in pbar:  # progress bar # batch -------------------------------------------------------
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # Forward
            with amp.autocast(enabled=cuda):  # stability issues when enabled
                loss = criterion(model(images), labels)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + " " * 36

                # Test
                if i == len(pbar) - 1:  # last batch
                    top1, top5, vloss = validate.run(
                        model=ema.ema, dataloader=validloader, criterion=criterion, pbar=pbar
                    )  # test accuracy, loss
                    fitness = top1  # define fitness as top1 accuracy
           # end batch ------------------------------------------------------------------------------------------------
        # Scheduler
        scheduler.step()
        stop = stopper(epoch=epoch, fitness=fitness)  # early stop check
        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness + opt.min_delta:
                best_fitness = fitness
            # Log
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss,
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "lr/0": optimizer.param_groups[0]["lr"],
            }  # learning rate
            logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "date": datetime.now().isoformat(),
                }
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt
        # EarlyStopping #TODO Early Stopping功能
        if stop:
            break  # must break all DDP ranks
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    # Train complete
    if RANK in {-1, 0} and final_epoch or stop:
        LOGGER.info(
            f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.",
            # f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
            f"\nResults saved to {colorstr('bold', save_dir)}"
            f'\nPredict:         python predict.py --weights {best} --source im.jpg'
            f'\nValidate:        python val.py --weights {best} --data {data_dir}'
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{best}')"
            f'\nVisualize:       https://netron.app\n'
        )

        # Plot examples
        images, labels = (x[:16] for x in next(iter(validloader)))  # first 16 images and labels
        pred = torch.max(ema.ema(images.to(device)), 1)[1]
        file = imshow_cls(images, labels, pred, de_parallel(model).names, verbose=False, f=save_dir / "valid_images.jpg")

        # Log results
        # meta = {"epochs": epochs, "top1_acc": best_fitness, "date": datetime.now().isoformat()}
        logger.log_images(file, name="Test Examples (true-predicted)", epoch=epoch)
        # logger.log_model(best, epochs, metadata=meta)

def parse_opt(known=False):
    """Parses command line arguments for YOLOv5 training including model path, dataset, epochs, and more, returning
    parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="initial weights path")
    parser.add_argument("--weights", nargs="+", type=str,default=ROOT / "runs/train-cls/exp35/weights/last.pt",
    help="model.pt path(s)")
    parser.add_argument("--cfg", type=str, default=ROOT /  "models/resnet18.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default="cifar10", help="cifar100, mnist, imagenet, ...")
    parser.add_argument("--epochs", type=int, default=10, help="total training epochs")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--batch-size", type=int, default=64, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=32, help="train, val image size (pixels)")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", nargs="?", const=True, default=False, help="start from i.e. --pretrained False")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="Adam", help="optimizer")
    parser.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--decay", type=float, default=5e-5, help="weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.01, help="EarlyStopping Minimum Delta (epochs without improvement)")
    parser.add_argument("--cutoff", type=int, default=None, help="Model layer cutoff index for Classify() head")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout (fraction)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    return parser.parse_known_args()[0] if known else parser.parse_args()
def main(opt):
    """Executes YOLOv5 training with given options, handling device setup and DDP mode; includes pre-training checks."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_git_status()
        # check_requirements(ROOT / "requirements.txt")

    device = select_device(opt.device, batch_size=opt.batch_size)

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)
def run(**kwargs):
    """
    Executes YOLOv5 model training or inference with specified parameters, returning updated options.

    Example: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)