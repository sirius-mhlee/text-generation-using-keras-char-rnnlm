import sys

import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

import DataOperator as do

import RNNLMNetwork as rn

def generate_sentence(char_to_index, index_to_char, char_count, model, input_char, generate_num):
    sentence = input_char.lower()

    for _ in range(generate_num):
        encoded = [char_to_index[char] for char in sentence]
        encoded = kr.preprocessing.sequence.pad_sequences([encoded], maxlen=cfg.max_sequence_len, padding='pre', truncating='pre')
        encoded = kr.utils.to_categorical(encoded, num_classes=char_count)

        predict_label_index = model.predict_classes(encoded)
        char = index_to_char[predict_label_index[0]]
        sentence = sentence + char

    return sentence

def main():
    input_model_path = sys.argv[1]
    input_char_to_index_path = sys.argv[2]
    input_char = sys.argv[3]
    input_predict_count = int(sys.argv[4])
    
    char_to_index = do.load_char_to_index(input_char_to_index_path)

    index_to_char = {}
    for char, index in char_to_index.items():
        index_to_char[index] = char

    char_count = len(char_to_index) + 1
    rnnlm_model = rn.create_model(char_count, input_model_path)

    print()
    print(generate_sentence(char_to_index, index_to_char, char_count, rnnlm_model, input_char, input_predict_count))

if __name__ == '__main__':
    main()