# Ratings Predictor Project
This project aims to take a review in natural English as an input, and to output a score between 1 and 10, corresponding to the rating that the reviewer would give, based on their review.

## Background
I wanted to pick a Natural Language Processing (NLP) task, because this is the field that I am preparing to work in, and because I find it fascinating. NLP models often focus on tasks such as 
* Question answering
* Sentiment analysis
* Machine translation
* Text summarisation
* Text generation
* Named entity recognition

However, I saw fewer examples of models that take natural language as input and output a scalar quantity, such as a rating from 1 to 10. I thought this would be an interesting challenge, because it was different to most of the NLP models I had seen, and yet seemed like it should, at least in theory, be fairly straightforward. I would be doing something very similar to sentiment analysis, but instead of classifying the data into discrete sentiments I would be giving it a numerical value. 

## Data
I decided to use a [dataset of IMDB movie ratings](http://ai.stanford.edu/~amaas/data/sentiment/) [[1]](#1). This dataset was designed for sentiment analysis - classifying the reviews as either positive or negative. For this reason it is split into two segments - positive reviews (rated 7 to 10) and negative reviews (rated 1 to 4). There are no 5 or 6 star reviews.

Ideally I would have had a dataset that was representative of all reviews, but I reasoned that this dataset would still contain a great deal of relevant information on which to train the model.

Future development of the model could benefit from a dataset that also contained some mid-rated reviews.

## Transfer learning
Since training an NLP model from scratch is prohibitively expensive in terms of time and computing power, I chose a base transformer ([distilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)) to process the text data. My model takes the output from this transformer, and trains a neural network to optimise for my target data - the rating for each review.

## Data processing
The various steps of data processing are outlined in the data preparation and analysis [notebook](https://github.com/woodwardmw/Ratings-predictor/blob/main/notebook.ipynb).

## Model training
I trained the model inside a Kaggle notebook, in order to make use of their GPUs. The experiments were logged using [ClearML](https://clear.ml), which I had never used before, but found to be very helpful for tuning hyperparameters.

The [training notebook](https://www.kaggle.com/markwoodward/mlzoomcamp-capstone-nlp-imdb-rating-predictor/settings?scriptVersionId=82297560) details how the model was trained, and the hyperparameters that I decided on.

## Local deployment with Flask
This repository includes a ```Pipfile``` and ```Pipfile.lock``` to define the virtual environment. The environment can be set up by running ```pipenv install```.

The local deployment can be set up by running ```predict.py```. It can then be tested with ```predict_test.py```, which contains a test string that can be modified.

## Building Docker container
The container can be built from the the [Dockerfile](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/blob/main/Dockerfile) by running:

```docker build -t ratings-predictor .```

and then run with:

```docker run -it --rm -p 9696:9696 ratings-predictor ```

## Deployment on AWS Lambda
I have also uploaded the Docker image to AWS Lambda, and created an API, which can be accessed at [https://ylvgq42zx5.execute-api.eu-west-2.amazonaws.com/test/predict](https://ylvgq42zx5.execute-api.eu-west-2.amazonaws.com/test/predict). The [lambda_test.py](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/blob/main/lambda_test.py) file runs a script to access this API. You can modify the requested text and run this file, to get ratings corresponding to your text review!

NOTE: It takes around 50 seconds for the AWS deployment to initiate. (This is because I had issues saving the complete model, and so could not then convert it to Tensorflow Lite, so it is running on full Tensorflow). So you need to run ```lambda_test.py```, which times out after about 30 seconds, and then wait another 20 seconds or so. After that, the AWS deployment should be ready, and will respond to the POST request in ```lambda_test.py``` in around 1 second.

## Examples
The model was mostly trained on long-ish (several hundred word) reviews, but it seems to work well on short reviews too.

![example](https://raw.githubusercontent.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/main/images/test%20examples/Screenshot%20from%202021-12-13%2021-28-03.png)

![example](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/raw/main/images/test%20examples/Screenshot%20from%202021-12-13%2021-40-38.png)

![example](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/raw/main/images/test%20examples/Screenshot%20from%202021-12-13%2021-28-18.png)

![example](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/raw/main/images/test%20examples/Screenshot%20from%202021-12-13%2021-28-56.png)

![example](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/raw/main/images/test%20examples/Screenshot%20from%202021-12-13%2021-29-47.png)


Although it was trained on movie data, there's no reason why the reviews need necessarily be restricted to movies...

![example](https://github.com/woodwardmw/MLZoomcamp-IMDB-ratings-predictor/raw/main/images/test%20examples/Screenshot%20from%202021-12-13%2021-31-01.png)

## References
<a id="1">[1]</a> 
Maas, A., Daly, R., Pham, P., Huang, D., Ng, A., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (pp. 142–150). Association for Computational Linguistics.
