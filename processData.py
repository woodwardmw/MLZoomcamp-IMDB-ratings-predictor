from datasets import load_dataset
import os

from datasets.filesystems import S3FileSystem
from botocore.session import Session

MAX_LENGTH = 512

def replace_char(char, replacement_char, dir):
    for dir_file in os.listdir(dir):
        with open(dir + '/' + dir_file, 'r') as file:
            filedata = file.read()
        # Replace the target string
        filedata = filedata.replace(char, replacement_char)

        # Write the file out again
        with open(dir + '/' + dir_file, 'w') as file:
            file.write(filedata)

def add_leading_zeros(dir):
    for dir_file in os.listdir(dir):
        file_number, file_end = dir_file.split('_')
        num = file_number.zfill(5)  # num is 5 characters long with leading 0
        new_file = "{}_{}".format(num, file_end)
        os.rename(dir + '/' + dir_file, dir + '/' + new_file)

def get_ordered_lists(files, split):
    files[split] = []
    subdirs = ['pos', 'neg']
    for subdir in subdirs:
        files_tmp = []
        for dir_file in os.listdir('data/aclImdb/' + split + '/' + subdir):
            files_tmp += [dir_file]
        files_tmp = sorted(files_tmp)
        files[split].extend(files_tmp)
    return files[split]

def get_ratings(files, ratings, split):
    ratings[split] = []
    for file in files[split]:
        rating = int(file.split('_')[1].split('.')[0])
        ratings[split].append(rating)
    return ratings[split]

def preprocess_data(dirs, splits):
    for dir in dirs:
        replace_char('\u0085', ' ', dir)  # Replace this character, which otherwise triggers a new row in the dataset
        add_leading_zeros(dir)  # Add leading zeros to files so alphabetical order is numerical order
    
    files = {}
    ratings = {}

    for split in splits:
        files[split] = get_ordered_lists(files, split)  # Put the files into ordered lists, one for train and one for test
        ratings[split] = get_ratings(files, ratings, split)  # Get two lists of the ratings, taken from the filenames
        
    data = load_dataset("text", data_files={'train': ['data/aclImdb/train/pos/*.txt', 'data/aclImdb/train/neg/*.txt'], 'test': ['data/aclImdb/test/pos/*.txt', 'data/aclImdb/test/neg/*.txt']})
    data['train'] = data['train'].add_column("rating", ratings['train'])
    data['test'] = data['test'].add_column("rating", ratings['test'])
    return data

def save_to_aws_bucket(data):
    # !mkdir -p ~/.aws
    # !cp ../aws_config ~/.aws/config
    s3_session = Session()
    s3 = S3FileSystem(session=s3_session)
    # s3.ls('mlzoomcamp-capstone')
    data.save_to_disk('s3://mlzoomcamp-capstone', fs=s3)

def tokenize_function(example, tokenizer, batched=True):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=MAX_LENGTH)

def data_subset(data, max_rows):
    return data.filter(lambda e, i: (i in range(int(max_rows / 2)) or i in range(int(len(data['train']) / 2), int(len(data['train']) / 2) + int(max_rows / 2))), with_indices=True)

def create_dataset(tokenized_data, split, data_collator, batch_size):
    return tokenized_data[split].to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            label_cols=["rating"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
        )