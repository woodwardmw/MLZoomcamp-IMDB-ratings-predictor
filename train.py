import model
import experiment
import processData
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TFDistilBertModel, DistilBertConfig
from functools import partial


DIRS = ['data/aclImdb/train/pos', 'data/aclImdb/train/neg', 'data/aclImdb/test/pos', 'data/aclImdb/test/neg']
SPLITS = ['train','test']

params = {'epochs': 15, 'lr': 0.0005, 'layer_dropout': 0, 'att_dropout': 0, 'max_length': 512, 'batch_size': 8, 'max_rows': 25000}

def main():
    data = processData.preprocess_data(DIRS, SPLITS)
    print(f"Data Processed: {data}")
    checkpoint = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    smaller_data = processData.data_subset(data, params['max_rows'])
    tokenized_data = smaller_data.map(partial(processData.tokenize_function, tokenizer=tokenizer, batched=True))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    tf_train_dataset = processData.create_dataset(tokenized_data, 'train', data_collator, params['batch_size'])
    tf_val_dataset = processData.create_dataset(tokenized_data, 'test', data_collator, params['batch_size']

    distilBERT = model.config_model(checkpoint, params)
    
    distilBERT_model = model.build_model(distilBERT, params)
    print(distilBERT_model.summary())

    experiment.run_experiment(distilBERT_model, tf_train_dataset, tf_val_dataset, params)

    distilBERT_model.save('my_model.h5')

if __name__ == '__main__':
    main()