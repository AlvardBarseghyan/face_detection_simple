## Face Detection - Simple

This project trains a computer to tell which class an image belongs to (for example, different kinds of faces). It uses Python and TensorFlow/Keras.

### What you need
- Python 3.9+ installed
- A folder of images organized by class (see below)

### Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Your data folder
Put images in folders like this:
```
data/
  Train/
    class_a/  (images for class A)
    class_b/
  Validate/
    class_a/
    class_b/
  Test/
    class_a/
    class_b/
```

### Train models
Run this from the project folder:
```bash
python main.py \
  --models fc cnn vgg \
  --train-dir data/Train \
  --valid-dir data/Validate \
  --test-dir  data/Test \
  --epochs 10 --batch-size 32 --lr 0.001 --augment
```

Results (charts, best weights, reports) will be saved in the `runs/` folder.

### Project structure
```
.
├── config.py                # Global configuration (paths, hyperparameters)
├── data/
│   └── dataloader.py        # Dataset preparation utilities (tf.data)
├── evaluation/
│   └── evaluate.py          # Evaluation and report generation
├── main.py                  # CLI for training/evaluation sessions
├── models/
│   ├── cnn_model.py         # Small CNN model
│   ├── fc_model.py          # Fully-connected baseline model
│   └── transfer_model.py    # VGG-based transfer learning model
├── training/
│   └── train.py             # Compile/train helpers, callbacks, TensorBoard
├── utils/
│   └── visualization.py     # Plotting: histories, confusion matrix
├── tests/
│   ├── test_dataloader.py   # Minimal dataset loader test
│   └── test_models.py       # Minimal model compile tests
└── requirements.txt
```


 