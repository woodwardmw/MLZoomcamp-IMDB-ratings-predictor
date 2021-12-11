FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.py", "./"]
RUN mkdir -p "models/distilbert-base-uncased"
COPY ["models/final_ratings_model_mse_4.054.h5", "./models"]
COPY ["models/distilbert-base-uncased/config.json", "models/distilbert-base-uncased/tf_model.h5", "models/distilbert-base-uncased/tokenizer.json", "models/distilbert-base-uncased/tokenizer_config.json", "models/distilbert-base-uncased/vocab.txt", "./models/distilbert-base-uncased/"]


#(For local deployment):
# EXPOSE 9696  
# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]  

ENTRYPOINT [ "gunicorn", "predict:app" ]