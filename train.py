import model
import experiment
import processData
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TFDistilBertModel, DistilBertConfig
from functools import partial
from clearml import Task, Logger



DIRS = ['data/aclImdb/train/pos', 'data/aclImdb/train/neg', 'data/aclImdb/test/pos', 'data/aclImdb/test/neg']
SPLITS = ['train','test']


params = {'epochs': 15, 'lr': 0.0005, 'layer_dropout': 0, 'att_dropout': 0, 'max_length': 512, 'batch_size': 8, 'max_rows': 25000}

def main():
    task = Task.init(project_name='IMDB Ratings Predict', task_name='Final model')
    task.connect(params)

    data = processData.preprocess_data(DIRS, SPLITS)
    print(f"Data Processed: {data}")
    checkpoint = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    smaller_data = processData.data_subset(data, params['max_rows'])
    tokenized_data = smaller_data.map(partial(processData.tokenize_function, tokenizer=tokenizer, batched=True))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    tf_train_dataset = processData.create_dataset(tokenized_data, 'train', data_collator, params['batch_size'])
    tf_val_dataset = processData.create_dataset(tokenized_data, 'test', data_collator, params['batch_size'])

    distilBERT = model.config_model(checkpoint, params)
    
    logger = task.get_logger()

    distilBERT_model = model.build_model(distilBERT, params)
    print(distilBERT_model.summary())

    train_history, trained_model = experiment.run_experiment(distilBERT_model, tf_train_dataset, tf_val_dataset, params)

    trained_model.save('my_model.h5')

    for iteration in range(params['epochs']):
        logger.report_scalar(title='MSE', series='provisional model, ' + str(params['max_rows']) + ' rows', value=train_history.history['val_mse'][iteration], iteration=iteration)

if __name__ == '__main__':
    main()