# FROM python:3.9-slim
FROM public.ecr.aws/lambda/python:3.9

RUN pip install pipenv

# WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY lambda_function.py .
COPY model.py .
# COPY __init__.py .

RUN mkdir -p models
COPY models/final_ratings_model_mse_4.054.h5 ./models
COPY ["models/distilbert-base-uncased/config.json", "models/distilbert-base-uncased/tf_model.h5", "models/distilbert-base-uncased/tokenizer.json", "models/distilbert-base-uncased/tokenizer_config.json", "models/distilbert-base-uncased/vocab.txt", "./models/distilbert-base-uncased/"]


#(For local deployment):
# EXPOSE 9696  
# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]  

# ENTRYPOINT [ "gunicorn", "predict:app" ]

CMD [ "lambda_function.lambda_handler" ]