import sys

import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

import DataOperator as do

import RNNLMNetwork as rn

def main():
    train_data_path = sys.argv[1]
    output_model_path = sys.argv[2]
    output_char_to_index_path = sys.argv[3]
    
    char_to_index, char_count, train_text, train_label = do.load_train_data(train_data_path)

    rnnlm_model = rn.create_model(char_count)
    rnnlm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    rnnlm_model.summary()

    rnnlm_model.fit(train_text, train_label, epochs=cfg.epochs)

    rnnlm_model.save_weights(output_model_path)
    do.save_char_to_index(char_to_index, output_char_to_index_path)

if __name__ == '__main__':
    main()