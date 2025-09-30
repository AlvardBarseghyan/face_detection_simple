"""Training utilities: default callbacks and compile/train helper."""

from typing import Sequence, Optional
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def get_default_callbacks(save_path: str, tb_log_dir: Optional[str] = None, use_tensorboard: bool = True):
    """Create a default set of callbacks for training.

    Includes ModelCheckpoint, EarlyStopping, and optional TensorBoard logging.
    """
    callbacks = [
        ModelCheckpoint(save_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    ]
    # this part one can add iteratively
    if use_tensorboard:
        if tb_log_dir is None:
            tb_log_dir = os.path.join(os.path.dirname(save_path), 'tb_logs')
        callbacks.append(TensorBoard(log_dir=tb_log_dir))
    return callbacks

def compile_and_train(model: tf.keras.Model, train_ds, valid_ds, epochs: int, lr: float, save_path: str, callbacks: Optional[Sequence]=None, tb_log_dir: Optional[str] = None):
    """Compile the model and run training returning the History object."""
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if callbacks is None:
        callbacks = get_default_callbacks(save_path, tb_log_dir)

    history = model.fit(train_ds, validation_data=valid_ds, epochs=epochs, callbacks=callbacks)
    return history


__all__ = [
    'get_default_callbacks',
    'compile_and_train',
]

