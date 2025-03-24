import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models
import torch.optim.lr_scheduler as lr_scheduler
from train import DatasetLoader, train_model
from test import test_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(num_classes, device):
    # Load model with weights
    weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
    net = models.resnext101_32x8d(weights=weights)
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.4),


        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(1024, num_classes)
    )

    num_params = count_parameters(net)
    print(f"Model has {num_params / 1e6:.2f}M parameters")

    return net.to(device)


def main():
    cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='Train ResNeXt '
                                                 'on a custom dataset')
    parser.add_argument('data_path',
                        type=str,
                        help='Root for the dataset.')
    parser.add_argument('--epochs', '-e',
                        type=int,
                        default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b',
                        type=int,
                        default=64,
                        help='Batch size.')
    parser.add_argument('--learning_rate', '-lr',
                        type=float,
                        default=5e-5,
                        help='Learning rate.')  # 0.00005
    parser.add_argument('--decay', '-d',
                        type=float,
                        default=3e-3,
                        help='Weight decay (L2 penalty).')  # 0.002
    parser.add_argument('--eta_min', '-em',
                        type=float,
                        default=1e-5,
                        help='Minimum learning rate for scheduler.')
    parser.add_argument('--save', '-s',
                        type=str,
                        default='./saved_models',
                        help='Folder to save checkpoints.')
    parser.add_argument('--mode', '-m',
                        type=str,
                        choices=['train', 'test'],
                        default='train',
                        help='Mode: train or test')
    args = parser.parse_args()

    print("CUDA Available:", torch.cuda.is_available())
    print("Current Device:", torch.cuda.current_device())
    print(
        "Using GPU:",
        torch.cuda.get_device_name(
            torch.cuda.current_device()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    dataset_loader = DatasetLoader(args.data_path, args.batch_size)
    train_loader, val_loader, num_classes = dataset_loader.load_data()

    net = build_model(num_classes=num_classes, device=device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.decay)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=args.eta_min)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    if (args.mode == 'train'):
        train_model(device=device,
                    net=net,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    scheduler=scheduler,
                    args=args)
    else:
        test_model(device=device,
                   net=net,
                   train_loader=train_loader,
                   args=args)


if __name__ == "__main__":
    main()
