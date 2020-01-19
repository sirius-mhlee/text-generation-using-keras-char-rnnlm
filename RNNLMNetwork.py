import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

def create_model(char_count, pre_train_file=None):
    model = kr.models.Sequential()

    model.add(kr.layers.LSTM(units=128, input_dim=char_count, input_length=cfg.max_sequence_len, name='lstm1'))
    model.add(kr.layers.Dense(units=char_count, activation='softmax', name='fc1'))

    if pre_train_file is not None:
        model.load_weights(pre_train_file, by_name=True)

    return model