import tensorflow as tf
from config import IMAGE_SIZE, NUM_CLASSES


def build_fc_model(input_shape=None):
    if input_shape is None:
        input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
            ])

