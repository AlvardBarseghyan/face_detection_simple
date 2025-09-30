from typing import Tuple, List
import os
import tensorflow as tf
from config import TRAIN_DIR, VALID_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, SEED


def set_seed(seed: int = 42):
	import random, numpy as np
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def prepare_dataset(path: str, batch_size: int = BATCH_SIZE, image_size: tuple = IMAGE_SIZE, augment: bool = False, shuffle: bool = True):
	"""Prepare a dataset from a directory.

	Args:
		path: Path to the dataset directory.
		batch_size: Batch size.
		image_size: Image size tuple.
		augment: Whether to apply augmentations to the training dataset.
		shuffle: Whether to shuffle the dataset.
	"""
	ds = tf.keras.utils.image_dataset_from_directory(
		path, image_size=image_size, shuffle=shuffle, seed=SEED, batch_size=batch_size
		)
	# Skip files that cannot be decoded (e.g., non-image or corrupted files)
	ds = ds.apply(tf.data.experimental.ignore_errors())
	# normalize and optimize pipeline
	augmentation = None
	if augment:
		augmentation = tf.keras.Sequential([
			tf.keras.layers.RandomFlip('horizontal'),
			tf.keras.layers.RandomRotation(0.1),
			tf.keras.layers.RandomZoom(0.1),
		])

	def _map_fn(x, y):
		x = tf.cast(x, tf.float32) / 255.0
		if augmentation is not None:
			x = augmentation(x, training=True)
		return x, y

	ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
	ds = ds.cache().prefetch(tf.data.AUTOTUNE)
	return ds


def load_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
	set_seed(SEED)
	train_ds = prepare_dataset(TRAIN_DIR, BATCH_SIZE, IMAGE_SIZE, augment=False)
	valid_ds = prepare_dataset(VALID_DIR, BATCH_SIZE, IMAGE_SIZE, augment=False)
	test_ds = prepare_dataset(TEST_DIR, BATCH_SIZE, IMAGE_SIZE, augment=False)
	return train_ds, valid_ds, test_ds


def get_class_names_from_dir(path: str) -> List[str]:
	entries = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
	return sorted(entries)


def load_datasets_from_dirs(train_dir: str, valid_dir: str, test_dir: str,
		batch_size: int = BATCH_SIZE, image_size: tuple = IMAGE_SIZE, augment: bool = False
	) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
	"""Load datasets from explicit directories and return class names.

	Args:
		train_dir: Directory containing class subfolders for training.
		valid_dir: Directory for validation data.
		test_dir: Directory for test data.
		batch_size: Batch size.
		image_size: Image size tuple.
		augment: Whether to apply augmentations to the training dataset.

	Returns:
		(train_ds, valid_ds, test_ds, class_names)
	"""
	set_seed(SEED)
	class_names = get_class_names_from_dir(train_dir)
	train_ds = prepare_dataset(train_dir, batch_size, image_size, augment=augment, shuffle=True)
	valid_ds = prepare_dataset(valid_dir, batch_size, image_size, augment=False, shuffle=False)
	test_ds = prepare_dataset(test_dir, batch_size, image_size, augment=False, shuffle=False)
	return train_ds, valid_ds, test_ds, class_names

