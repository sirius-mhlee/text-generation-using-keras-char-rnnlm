import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

def load_char_to_index(char_to_index_path):
  with open(char_to_index_path, 'rb') as char_to_index_file:
      return pickle.load(char_to_index_file)

def save_char_to_index(char_to_index, char_to_index_path):
  with open(char_to_index_path, 'wb') as char_to_index_file:
      pickle.dump(char_to_index, char_to_index_file, protocol=pickle.HIGHEST_PROTOCOL)

def load_train_data(train_data_path):
    train_data_file = open(train_data_path, 'r')
    train_data_text = train_data_file.read()
    train_data_file.close()
    train_data_text = train_data_text.lower().replace('\r', '').replace('\n', ' ')

    char_list = sorted(list(set(train_data_text)))
    char_count = len(char_list) + 1
    char_to_index = dict((char, index + 1) for index, char in enumerate(char_list))
    encoded_train_data_text = [char_to_index[char] for char in train_data_text]

    sequence_list = list()
    for pos in range(cfg.max_sequence_len + 1, len(encoded_train_data_text)):
        encoded = encoded_train_data_text[pos - (cfg.max_sequence_len + 1):pos]
        for i in range(1, len(encoded)):
            sequence_list.append(encoded[:i + 1])

    sequence_list = kr.preprocessing.sequence.pad_sequences(sequence_list, maxlen=cfg.max_sequence_len + 1, padding='pre', truncating='pre')
    sequence_list = np.array(sequence_list)

    train_text = kr.utils.to_categorical(sequence_list[:, :-1], num_classes=char_count)
    train_label = kr.utils.to_categorical(sequence_list[:, -1], num_classes=char_count)

    return char_to_index, char_count, train_text, train_label