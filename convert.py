import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel

from model import build_model

def convert_model(model_dir):
    model = tf.saved_model.load(model_dir)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('model.tflite', 'wb') as f:
        f.write(tflite_model)


# params = {'epochs': 15, 'lr': 0.0005, 'layer_dropout': 0, 'att_dropout': 0, 'max_length': 512, 'batch_size': 8, 'max_rows': 25000}

# distilBERT = TFDistilBertModel.from_pretrained('./models/distilbert-base-uncased', local_files_only=True)
# # model = build_model(distilBERT, params=params)
# # model.load_weights('provisional_model.h5')


# tf.saved_model.save(distilBERT, 'models')

def main():
    convert_model('models/final_ratings_model.h5')

if __name__ == '__main__':
    main()