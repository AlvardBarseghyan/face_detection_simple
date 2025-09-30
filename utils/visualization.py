import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Any, List, Optional


def plot_history(history: Any, title_prefix: str = ""):
    epochs = np.arange(1, len(history.history['accuracy']) + 1)

    plt.figure()
    plt.plot(epochs, history.history['accuracy'], marker='o', label='Training')
    plt.plot(epochs, history.history['val_accuracy'], marker='o', label='Validation')
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(epochs, history.history['loss'], marker='o', label='Training')
    plt.plot(epochs, history.history['val_loss'], marker='o', label='Validation')
    plt.title(f"{title_prefix} Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


def save_history_plots(history: Any, output_dir: str, title_prefix: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)
    epochs = np.arange(1, len(history.history['accuracy']) + 1)

    plt.figure()
    plt.plot(epochs, history.history['accuracy'], marker='o', label='Training')
    plt.plot(epochs, history.history['val_accuracy'], marker='o', label='Validation')
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, history.history['loss'], marker='o', label='Training')
    plt.plot(epochs, history.history['val_loss'], marker='o', label='Validation')
    plt.title(f"{title_prefix} Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()


def plot_and_save_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: str, normalize: bool = True, title: Optional[str] = None) -> None:
    if normalize:
        with np.errstate(all='ignore'):
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_title(title or 'Confusion matrix')
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
