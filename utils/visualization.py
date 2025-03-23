import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
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


def plot_loss_accuracy(train_losses, val_losses, train_accuracies,
                       val_accuracies, save_fig=True,
                       output_path="training_curve.png"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
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
