from flask import Flask, request, jsonify

from transformers import DistilBertTokenizer, TFDistilBertModel
from model import build_model

checkpoint = 'distilbert-base-uncased' 
# checkpoint = './models/distilbert-base-uncased' 
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
params = {'epochs': 15, 'lr': 0.0005, 'layer_dropout': 0, 'att_dropout': 0, 'max_length': 512, 'batch_size': 8, 'max_rows': 25000}
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
distilBERT = TFDistilBertModel.from_pretrained(checkpoint)
# distilBERT = model.config_model(checkpoint)
model = build_model(distilBERT, params=params)
model.summary() # Remove later
# model.load_weights('models/final_ratings_model.h5')
model.load_weights('models/final_ratings_model_mse_4.054.h5')


app = Flask('vowel')

@app.route('/welcome', methods=['GET'])
def welcome():
    welcome_msg = "<h1>Welcome to your application deployed as Docker container on heroku</h1>"
    return welcome_msg

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    tensor = tokenizer(text, truncation=True, padding='max_length', max_length=params['max_length'], return_tensors='tf')
    prediction = model.predict(tensor.data)[0][0]
    result = {'Prediction': str(prediction)}
    return jsonify(result)



# convert_model('models')  # I can't get this to work, because tensorflow doesn't save models with custom layers in a way that can be converted to TF Lite.
# new_model = tf.keras.models.load_model('new_model')

# # print(model.summary())
# string = """
# This is a pale imitation of 'Officer and a Gentleman.' There is NO chemistry between Kutcher and the unknown woman who plays his love interest. The dialog is wooden, the situations hackneyed. It's too long and the climax is anti-climactic(!). I love the USCG, its men and women are fearless and tough. The action scenes are awesome, but this movie doesn't do much for recruiting, I fear. The script is formulaic, but confusing. Kutcher's character is trying to redeem himself for an accident that wasn't his fault? Costner's is raging against the dying of the light, but why? His 'conflict' with his wife is about as deep as a mud puddle. I saw this sneak preview for free and certainly felt I got my money's worth.
# """
# tensor = tokenizer(string, truncation=True, padding='max_length', max_length=params['max_length'], return_tensors='tf')
# prediction = model.predict(tensor.data)
# print(prediction)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)
