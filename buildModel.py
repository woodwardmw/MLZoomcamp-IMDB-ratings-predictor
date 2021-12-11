from transformers import AutoTokenizer, DataCollatorWithPadding
import model
import tensorflow as tf

checkpoint = 'distilbert-base-uncased'  # Would be better to have local versions of these!
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

distilBERT = model.config_model(checkpoint)
model = model.build_model(distilBERT)
model.summary() # Remove later
model.load_weights('provisional_model.h5')
tf.keras.models.save_model(model, 'new_model', save_format = 'h5')