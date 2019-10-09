import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor

from train2 import train
from data_handler import MyCoco
from models2 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images",
                        default=os.path.join("..", "data", "val2017"),
                        help="The relative path to the validation data directory")
    parser.add_argument("--train_annotation",
                        default=os.path.join("..", "data", "annotations", "instances_val2017.json"),
                        help="The relative path to the validation data directory")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs to train")
    parser.add_argument("--iterations", type=int, default=16,
                        help="Number of iterations for each epoch")
    parser.add_argument("--noise_factor", type=float, default=0.0,
                        help="Number of epochs to train")
    parser.add_argument("--model_name", type=str, default="conv_rnn",
                        help="Weighting of L1 Loss")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning Rate")
    parser.add_argument("--model_type", type=str, default="conv_rnn",
                        help="Model type")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic binarizer")
    parser.add_argument("--size", type=int, default=256,
                            help="Spatial width and height of image")

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print(f"device: {args.device}")

    train_params = {
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': 4,
        'pin_memory': False,
        'iterations': args.iterations,
        'validate': True
    }

    print(f"ARGUMENTS: {args}\n")
    print(f"TRAIN PARAMS: {train_params}\n")

    target_transform = transforms.Compose([Resize((args.size, args.size)), ToTensor()])
    input_transform = transforms.Compose([Resize((args.size, args.size)), ToTensor()])

    train_dataset_og = MyCoco(
        root=args.train_images,
        annFile=args.train_annotation,
        noise_factor=args.noise_factor,
        input_transform=input_transform,
        target_transform=target_transform
    )

    lengths = (int(0.1*len(train_dataset_og)), int(0.9*len(train_dataset_og)))
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset_og, lengths)
    print(f"Train dataset length: {len(train_dataset)}")
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=train_params['batch_size'],
        pin_memory=train_params['pin_memory'])
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=train_params['batch_size'],
        pin_memory=train_params['pin_memory'])
    train(train_params, args, train_loader, val_loader)
