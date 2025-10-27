"""CLI entry point to train and evaluate face detection models.

Provides utilities to build models, run a single training/evaluation session,
and orchestrate multiple runs via command-line arguments.
"""

import argparse
import datetime as dt
import json
import os
from typing import Dict

from data.dataloader import load_datasets, get_class_names_from_dir
from models.fc_model import build_fc_model
from models.cnn_model import build_cnn_model
from models.transfer_model import build_vgg_model
from training.train import compile_and_train
from evaluation import evaluate_and_report
from utils.visualization import save_history_plots, plot_and_save_confusion_matrix
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, TRAIN_DIR, VALID_DIR, TEST_DIR


def build_model(kind: str):
    """Return a model instance by kind: 'fc', 'cnn', or 'vgg'."""
    kind = kind.lower()
    if kind == 'fc':
        return build_fc_model()
    if kind == 'cnn':
        return build_cnn_model()
    if kind in ('vgg', 'transfer'):
        return build_vgg_model(freeze_backbone=True)
    raise ValueError(f"Unknown model kind: {kind}")


def run_one(kind: str, run_dir: str, train_dir: str, valid_dir: str, test_dir: str,
            epochs: int, batch_size: int, lr: float, augment: bool = False) -> Dict:
    """Train and evaluate one model variant, saving artifacts under run_dir.

    Returns a dict with metrics and paths to saved reports.
    """
    os.makedirs(run_dir, exist_ok=True)
    # Save config
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'model': kind,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'train_dir': train_dir,
            'valid_dir': valid_dir,
            'test_dir': test_dir,
            'augment': augment
        }, f, indent=2)

    # Data
    class_names = get_class_names_from_dir(train_dir)
    train_ds, valid_ds, test_ds = load_datasets(
        train_dir, valid_dir, test_dir, batch_size=batch_size, augment=augment
    )

    # Model and training
    model = build_model(kind)
    weights_path = os.path.join(run_dir, 'best.h5')
    history = compile_and_train(model, train_ds, valid_ds, epochs, lr, save_path=weights_path,
                                tb_log_dir=os.path.join(run_dir, 'tb_logs'))

    # Save training curves
    save_history_plots(history, output_dir=os.path.join(run_dir, 'plots'), title_prefix=kind.upper())

    # Evaluate
    eval_dir = os.path.join(run_dir, 'eval')
    results = evaluate_and_report(model, test_ds, class_names, eval_dir)
    plot_and_save_confusion_matrix(results['confusion_matrix'], class_names,
                                   output_path=os.path.join(eval_dir, 'confusion_matrix.png'),
                                   normalize=True, title=f"{kind.upper()} Confusion Matrix")
    return results


def main():
    """Parse CLI arguments and run training/evaluation for requested models."""
    parser = argparse.ArgumentParser(description='Train face detection models')
    parser.add_argument('--models', nargs='+', default=['fc', 'cnn', 'vgg'], help='Models to train')
    parser.add_argument('--train-dir', default=TRAIN_DIR)
    parser.add_argument('--valid-dir', default=VALID_DIR)
    parser.add_argument('--test-dir', default=TEST_DIR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--augment', action='store_true', help='Enable training-time augmentations')
    parser.add_argument('--runs-dir', default='runs', help='Base directory to store runs')
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(args.runs_dir, f'run_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)

    index = {}
    for kind in args.models:
        model_dir = os.path.join(session_dir, kind)
        res = run_one(kind, model_dir, args.train_dir, args.valid_dir, args.test_dir,
                      args.epochs, args.batch_size, args.lr, augment=args.augment)
        index[kind] = {
            'run_dir': model_dir,
            'metrics_path': os.path.join(model_dir, 'eval', 'metrics.json')
        }

    with open(os.path.join(session_dir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)


if __name__ == '__main__':
    main()

