"""visualization.py - Contains utilities for plotting confusion matrix and training curves."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    """
    Plot and save the confusion matrix as a heatmap.

    Args:
        true_labels (list or array): Ground truth labels.
        pred_labels (list or array): Predicted labels from the model.
        class_names (list): List of class names.
        save_path (str): File path to save the plot.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    step = max(len(class_names) // 50, 1)
    ax.set_xticks(ax.get_xticks()[::step])
    ax.set_yticks(ax.get_yticks()[::step])
    ax.set_xticklabels(class_names[::step], rotation=45, ha="right")
    ax.set_yticklabels(class_names[::step], rotation=0)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Confusion matrix saved at {save_path}")
    plt.close()


def plot_loss_accuracy(train_losses, val_losses,
                       train_accuracies, val_accuracies,
                       save_fig=True, output_path="training_curve.png"):
    """
    Plot and optionally save training/validation loss and accuracy curves.

    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        train_accuracies (list): Training accuracy per epoch.
        val_accuracies (list): Validation accuracy per epoch.
        save_fig (bool): Whether to save the figure.
        output_path (str): Path to save the figure if save_fig is True.
    """
    _, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot loss curve
    ax[0].plot(
        range(
            len(train_losses)),
        train_losses,
        label="Train Loss",
        marker='o')
    ax[0].plot(
        range(
            len(val_losses)),
        val_losses,
        label="Validation Loss",
        marker='o')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training & Validation Loss")
    ax[0].legend()
    ax[0].grid()

    # Plot accuracy curve
    ax[1].plot(
        range(
            len(train_accuracies)),
        train_accuracies,
        label="Train Accuracy",
        marker='o')
    ax[1].plot(
        range(
            len(val_accuracies)),
        val_accuracies,
        label="Validation Accuracy",
        marker='o')
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("Training & Validation Accuracy")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path)
        print(f"Training curve saved as {output_path}")
    plt.close()
