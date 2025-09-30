"""Evaluation utilities: inference over datasets and metric/report generation."""

import json
import os
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model: tf.keras.Model, test_ds) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on test dataset and return true and predicted labels."""
    y_true_list = []
    y_pred_list = []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_pred_list.append(pred_labels)
        y_true_list.append(batch_y.numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return y_true, y_pred


def evaluate_and_report(model: tf.keras.Model, test_ds, class_names: Sequence[str], output_dir: str) -> Dict[str, Any]:
    """Evaluate model, generate metrics and report files, and return results dict."""
    os.makedirs(output_dir, exist_ok=True)
    y_true, y_pred = evaluate_model(model, test_ds)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Save report JSON
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Save metrics summary
    metrics = {
        'accuracy': float(report.get('accuracy', 0.0)),
        'macro_avg_f1': float(report.get('macro avg', {}).get('f1-score', 0.0)),
        'weighted_avg_f1': float(report.get('weighted avg', {}).get('f1-score', 0.0)),
    }
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'metrics': metrics,
        'classification_report_path': report_path,
        'metrics_path': metrics_path,
    }


