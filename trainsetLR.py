import os
import argparse
import random
import shutil
import sys
import math


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

from src.datasets import ImageFolder
from src.ops import compute_padding
from src.models import image_models


from torch.utils.tensorboard import SummaryWriter
import logging
from typing import Tuple, Union, List
import numpy as np

from pytorch_msssim import ms_ssim
import PIL
import PIL.Image as Image
from pathlib import Path
from tqdm import tqdm

from datetime import datetime



def adjust_LR(optimizer,epoch):
    if epoch < 85:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
    elif epoch < 125:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 3e-5
    elif epoch < 140:           
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-6


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metrics = metrics

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metrics == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == 'ms-ssim':
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]
        return out


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())

def read_image(filepath: str) -> torch.Tensor:
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)
def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)



def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.learning_rate,
    )
    return optimizer, aux_optimizer



def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm,logger_train, tb_logger, current_step
):
    model.train()
    device = next(model.parameters()).device

    for i, d in tqdm(enumerate(train_dataloader), total=len(train_dataloader),smoothing=0.9,desc="Epoch: %d" % epoch):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('train/loss'), out_criterion["loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('train/bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            if out_criterion["mse_loss"] is not None:
                tb_logger.add_scalar('{}'.format('train/mse_loss'), out_criterion["mse_loss"].item(), current_step)
            if out_criterion["ms_ssim_loss"] is not None:
                tb_logger.add_scalar('{}'.format('train/ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)

        if i % 100 == 0:
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            
    return current_step

def test_epoch(epoch, test_dataloader, model, criterion, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()


    with torch.no_grad():
        for i, d in tqdm(enumerate(test_dataloader), total=len(test_dataloader),smoothing=0.9,desc="Epoch: %d" % epoch):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])



    tb_logger.add_scalar('{}'.format('val/loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('val/bpp_loss'), bpp_loss.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.5f} | "
            f"MSE loss: {mse_loss.avg:.5f} | "
            f"Bpp loss: {bpp_loss.avg:.3f} | "
            f"Aux loss: {aux_loss.avg:.2f} "

        )
        tb_logger.add_scalar('{}'.format('val/mse_loss'), mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.5f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.5f} | "
            f"Bpp loss: {bpp_loss.avg:.3f} | "
            f"Aux loss: {aux_loss.avg:.2f} "
        )
        tb_logger.add_scalar('{}'.format('val/ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)

    return loss.avg

def valid_epoch(epoch,valid_path, model, criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    dataset_name = valid_path.split(os.sep)[-1]
    with torch.no_grad():
        for i, file_path in enumerate(collect_images(valid_path)):
            d = read_image(file_path).to(device).unsqueeze(0)
            h, w = d.size(2), d.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            d = F.pad(d, pad, mode="constant", value=0)
                    
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

            rec = torch2img(F.pad(out_net['x_hat'],unpad))
            img = torch2img(F.pad(d,unpad))
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(os.path.join(save_dir,dataset_name)):
                os.makedirs(os.path.join(save_dir,dataset_name))
            rec.save(os.path.join(save_dir,dataset_name ,'%03d_rec.png' % i))
            img.save(os.path.join(save_dir, dataset_name,'%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format(dataset_name+'/loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(dataset_name+'/bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(dataset_name+'/psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(dataset_name+'/ms-ssim'), ms_ssim.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"==Valid dataset {dataset_name}=="
            f"Valid epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.5f} | "
            f"MSE loss: {mse_loss.avg:.5f} | "
            f"Bpp loss: {bpp_loss.avg:.3f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format(dataset_name+'/mse_loss'), mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"==Valid dataset {dataset_name}=="
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.5f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.5f} | "
            f"Bpp loss: {bpp_loss.avg:.3f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format(dataset_name+'/ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        dest_filename = filename.replace(filename.split(os.sep)[-1], "checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, dest_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=1,
        help="Quality (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=16,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="checkpoint model path"
    )
    parser.add_argument(
        "-p",
        "--pretrain",
        default=None,
        type=str,
        help="pretrained model path"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))

    setup_logger('train', os.path.join('./experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', os.path.join('./experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)
    
    logger_train.info(', '.join(f'{k}={v}' for k, v in vars(args).items()))
    if not os.path.exists(os.path.join('./experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('./experiments', args.experiment, 'checkpoints'))



    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = image_models[args.model](quality=args.quality)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)

    criterion = RateDistortionLoss(lmbda=args.lmbda)

    
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        best_loss = checkpoint['loss']
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

        current_step = checkpoint["epoch"] * math.ceil(len(train_dataloader.dataset) / args.batch_size)
    else:
        last_epoch = 0
        best_loss = float("inf")
        current_step = 0
    
    if args.pretrain:
        print("Loading", args.pretrain)
        checkpoint = torch.load(args.pretrain, map_location=device)        
        net.load_state_dict(checkpoint["state_dict"])
        
    for epoch in range(last_epoch, args.epochs):
        adjust_LR(optimizer,epoch)
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        tb_logger.add_scalar('{}'.format('train/LearningRate'), optimizer.param_groups[0]['lr'], epoch + 1)
        current_step =train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step
        )
        
        loss = test_epoch(epoch, test_dataloader, net, criterion, logger_val, tb_logger)

        save_dir = os.path.join('./experiments', args.experiment, 'val_images', '%03d' % (epoch + 1))
        valid_epoch(epoch, args.dataset+os.sep+"Kodak", net, criterion,save_dir, logger_val, tb_logger)
        valid_epoch(epoch, args.dataset+os.sep+"CLIC", net, criterion,save_dir, logger_val, tb_logger)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                },
                is_best,
                os.path.join('./experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_val.info('best checkpoint saved.')

if __name__ == "__main__":
    main(sys.argv[1:])