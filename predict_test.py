import requests

example = """
This was pretty good, although not amazing. I liked it though!
"""

example_dict = {'text': example}

# url = 'http://localhost:9696/predict'
url = 'https://imdb-ratings-predictor.herokuapp.com/predict'
response = requests.post(url, json=example_dict)
result = response.json()
print(f"Predicted rating: {min(float(result['Prediction']), 10):.2f}")