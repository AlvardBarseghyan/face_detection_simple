import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from config import IMAGE_SIZE, NUM_CLASSES


def build_vgg_model(freeze_backbone: bool = True):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    vgg.trainable = not freeze_backbone

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        vgg,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
    return model


def get_transfer_callbacks(filepath: str = "transfer_model.h5"):
    return [ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
            ]
