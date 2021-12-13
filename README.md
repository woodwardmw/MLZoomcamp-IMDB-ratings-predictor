# Ratings Predictor Project
This project aims to take a review in natural English as an input, and to output a score between 1 and 10, corresponding to the rating that the reviewer would give, based on their review.

## Background
I wanted to pick a Natural Language Processing (NLP) task, because this is the field that I am preparing to work in, and because I find it fascinating. NLP models often focus on a number of different tasks, such as 
* List tasks
* etc

However, I saw very few examples of models that take natural language as input and output a scalar quantity, such as a rating from 1 to 10. I thought this would be an interesting challenge, because it was different to most of the NLP models I had seen, and yet seemed like it should, at least in theory, be possible.

## Data
I decided to us a [dataset of IMDB movie ratings](http://ai.stanford.edu/~amaas/data/sentiment/). This dataset was designed for sentiment analysis - classifying the reviews as either positive or negative. For this reason it is split into two segments - positive reviews (rated 7 to 10) and negative reviews (rated 1 to 4). There are no 5 or 6 star reviews.

Ideally I would have had a dataset that was representative of all reviews, but I reasoned that this dataset would still contain a great deal of relevant information on which to train the model.

Future development of the model could benefit from a dataset that also contained some mid-rated reviews.

## Transfer learning
Since training an NLP model from scratch is prohibitively expensive in terms of time and computing power, I chose a base transformer ([distilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)) to process the text data. My model takes the output from this transformer, and trains a neural network to optimise for my target data - the rating for each review.

## Data processing
The various steps of data processing are outlined in the [Data preparation and analysis](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/blob/main/Ratings%20Predictor%20Data%20Preparation%20and%20Analysis.ipynb) notebook.

## Model training
I trained the model inside a Kaggle notebook, in order to make use of their GPUs. The experiments were logged using [ClearML](https://clear.ml), which I had never used before, but found to be very helpful for tuning hyperparameters.

The [training notebook](https://www.kaggle.com/markwoodward/mlzoomcamp-capstone-nlp-imdb-rating-predictor?scriptVersionId=82290969) details how the model was trained, and the hyperparameters that I decided on.

## Local deployment with Flask
The local deployment can be set up by running ```predict.py```. It can then be tested with ```predict_test.py```, which contains a test string that can be modified.

## Deployment on AWS Lambda
The [Dockerfile](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/blob/main/Dockerfile) creates a Docker image that I have uploaded to AWS Lambda, and created an API, which can be accessed at [https://ylvgq42zx5.execute-api.eu-west-2.amazonaws.com/test/predict](https://ylvgq42zx5.execute-api.eu-west-2.amazonaws.com/test/predict). The [lambda_test.py](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/blob/main/lambda_test.py) file runs a script to access this API. You can modify the requested text and run this file, to get ratings corresponding to your text review!
