import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "F:/Python/ML/Machine-Learning-Collection/ML/Pytorch\GANs/CycleGAN\data/train"
VAL_DIR = "F:/Python/ML/Machine-Learning-Collection/ML/Pytorch/GANs/CycleGAN\data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = True
SAVE_MODEL = True
SAVE_DIR="F:/Python/ML/Machine-Learning-Collection/ML/Pytorch/GANs/CycleGAN/saved_images"
CHECKPOINT_GEN_H = "F:/Python/ML/Machine-Learning-Collection/ML/Pytorch/GANs/CycleGAN/CycleGAN_weights/genh.pth.tar"
CHECKPOINT_GEN_Z = "F:/Python/ML/Machine-Learning-Collection/ML/Pytorch/GANs/CycleGAN/CycleGAN_weights/genz.pth.tar"
CHECKPOINT_CRITIC_H = "F:/Python/ML/Machine-Learning-Collection/ML/Pytorch/GANs/CycleGAN/CycleGAN_weights/critich.pth.tar"
CHECKPOINT_CRITIC_Z = "F:/Python/ML/Machine-Learning-Collection/ML/Pytorch/GANs/CycleGAN/CycleGAN_weights/criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)