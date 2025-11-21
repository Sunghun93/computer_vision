from pl_bolts.datamodules import CIFAR10DataModule
from torch.optim.lr_scheduler import OneCycleLR
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl
from ViT import ViT
from torchmetrics.functional import accuracy
import torch
import warnings
warnings.filterwarnings("ignore")

config = {
    "data_dir": ".",
    "batch_size": 256,
    "num_workers": 2,
    "num_classes": 10,
    "lr": 1e-4,
    "max_lr": 1e-3
}

train_transforms = T.Compose(
    [
        T.RandomCrop(32, padding=4),  # perturbation
        T.RandomHorizontalFlip(),  # perturbation
        T.ToTensor(),
        cifar10_normalization()
    ]
)

test_transforms = T.Compose(
    [
        T.ToTensor(),
        cifar10_normalization()
    ]
)

# Train: 45,000, valid: 5,000, test: 10,000
cifar10_dm = CIFAR10DataModule(
    data_dir=config["data_dir"],
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms
)