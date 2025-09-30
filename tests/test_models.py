from models.fc_model import build_fc_model
from models.cnn_model import build_cnn_model
from models.transfer_model import build_vgg_model


def test_models_compile():
    m1 = build_fc_model()
    m1.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    m2 = build_cnn_model()
    m2.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    m3 = build_vgg_model()
    m3.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    assert True

# usage:
# pytest tests/test_models.py