"""Training pipeline for ResNeXt with data augmentation, Focal Loss, EMA, and scheduler support."""

import os
import gc
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch_ema import ExponentialMovingAverage
from sklearn.utils.class_weight import compute_class_weight

from utils import plot_confusion_matrix, plot_loss_accuracy, clear_memory
from utils import SmoothFocalLoss, FocalLoss


class DatasetLoader:
    """Data loader for training and validation datasets with augmentation."""

    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.15, 0.15),
                    scale=(0.7, 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                    hue=0.125),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                ], p=0.3),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        }

    def load_data(self):
        """
        Loads training and validation datasets using ImageFolder and returns their data loaders.

        Returns:
            tuple:
                - DataLoader: Training data loader.
                - DataLoader: Validation data loader.
                - int: Number of classes.
        """
        train_data = ImageFolder(
            os.path.join(
                self.data_path,
                "train/"),
            transform=self.transform['train'])
        val_data = ImageFolder(
            os.path.join(
                self.data_path,
                "val/"),
            transform=self.transform['val'])
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2)
        val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2)
        num_classes = len(train_data.classes)
        return train_loader, val_loader, num_classes


class Trainer:
    """Handles model training, validation, checkpointing, and EMA management."""

    def __init__(self, model, train_loader, val_loader,
                 optimizer, criterion, device, save_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.best_acc = 0.0
        self.scaler = torch.amp.GradScaler(device="cuda")
        os.makedirs(save_path, exist_ok=True)

    def train(self, epoch):
        """
        Performs one epoch of model training.

        Args:
            epoch (int): Current training epoch number.

        Returns:
            tuple: Training loss and accuracy for the current epoch.
        """
        self.model.train()
        gc.collect()
        total_loss, correct, total = 0.0, 0, 0
        print(f"\nEpoch {epoch+1}: Training...")

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device).long().view(-1)
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                output = self.model(images)
                loss = self.criterion(output, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx+1}/{len(self.train_loader)} \
                    | Loss: {loss.item():.4f}")

        train_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        print(
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        return train_loss, train_acc

    def validate(self, epoch, confusion_matrix_folder):
        """
        Evaluates the model on the validation set.

        Args:
            epoch (int): Current epoch number.
            confusion_matrix_folder (str): Directory to save the confusion matrix plot.

        Returns:
            tuple: Validation loss and accuracy.
        """
        self.model.eval()
        gc.collect()
        correct, total, total_loss = 0, 0, 0.0
        true_labels, pred_labels = [], []
        print("\nValidating...")
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                total_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())

        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        print(
            f"Validation Loss: {val_loss:.4f} \
            | Validation Accuracy: {val_acc:.2f}%")

        class_names = self.val_loader.dataset.classes
        output_path = os.path.join(
            confusion_matrix_folder,
            f"confusion_matrix_epoch_{epoch + 1}.png")
        plot_confusion_matrix(
            true_labels,
            pred_labels,
            class_names,
            output_path)

        return val_loss, val_acc

    def save_model(self, epoch, val_acc, train_losses,
                   train_accuracies, val_losses, val_accuracies, ema):
        """
        Saves model checkpoint and best-performing model if applicable.

        Args:
            epoch (int): Current epoch number.
            val_acc (float): Current validation accuracy.
            train_losses (list): History of training losses.
            train_accuracies (list): History of training accuracies.
            val_losses (list): History of validation losses.
            val_accuracies (list): History of validation accuracies.
            ema (ExponentialMovingAverage): EMA object to save.
        """
        checkpoint_path = os.path.join(self.save_path, 'latest_checkpoint.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'ema': ema.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Latest checkpoint saved at: {checkpoint_path}")

        if val_acc >= self.best_acc:
            self.best_acc = val_acc
            best_model_path = os.path.join(
                self.save_path, f'best_model.pth')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {self.best_acc:.2f}%")

            torch.save(
                ema.state_dict(),
                os.path.join(
                    self.save_path,
                    f'ema_state.pth'))
            print("EMA state saved.")


def train_model(device, net, optimizer, train_loader,
                val_loader, criterion, scheduler, args):
    """
    Main training loop.
    Handles early stopping, EMA, progressive loss switching, and checkpointing.
    """
    confusion_matrix_folder = "confusion_matrices"
    os.makedirs(confusion_matrix_folder, exist_ok=True)

    best_acc = 0
    start_epoch = 0
    checkpoint_path = os.path.join(args.save, 'latest_checkpoint.pth')

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # early_stopping = EarlyStopping(patience=10, delta=0.001)
    ema = ExponentialMovingAverage(net.parameters(), decay=0.999)

    # === Compute class weights ===
    all_targets = []
    for _, labels in train_loader:
        all_targets.extend(labels.numpy())
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(train_loader.dataset.classes)),
        y=all_targets
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        train_accuracies = checkpoint['train_accuracies']
        train_losses = checkpoint['train_losses']
        val_accuracies = checkpoint['val_accuracies']
        val_losses = checkpoint['val_losses']
        ema.load_state_dict(checkpoint['ema'])
        print(
            f"Resuming training from epoch {start_epoch},\
              best accuracy: {best_acc:.2f}%")

    trainer = Trainer(
        net,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        args.save)
    trainer.best_acc = best_acc

    for epoch in range(start_epoch, args.epochs):
        # === Progressive loss switching ===
        if 14 <= epoch < 17:
            alpha = (epoch - 14) / 3  # 0, 0.33, 0.66
            trainer.criterion = SmoothFocalLoss(
                gamma=1.5, smoothing=0.05, alpha=alpha, weight=class_weights)
            trainer.scaler = torch.amp.GradScaler(device="cuda")
        elif epoch == 17:
            trainer.criterion = FocalLoss(gamma=1.5, weight=class_weights)
            trainer.scaler = torch.amp.GradScaler(device="cuda")

        train_loss, train_acc = trainer.train(epoch)
        ema.update()

        # === Delay EMA effect until epoch 20 ===
        if epoch < 20:
            val_loss, val_acc = trainer.validate(
                epoch, confusion_matrix_folder)
        else:
            with ema.average_parameters():
                val_loss, val_acc = trainer.validate(
                    epoch, confusion_matrix_folder)

        scheduler.step()
        clear_memory()

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        trainer.save_model(
            epoch,
            val_acc,
            train_losses,
            train_accuracies,
            val_losses,
            val_accuracies,
            ema)

    plot_loss_accuracy(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        save_fig=True,
        output_path="train_curve.png")
