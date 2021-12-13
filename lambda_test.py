import requests

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://ylvgq42zx5.execute-api.eu-west-2.amazonaws.com/test/predict'
data = {'text': """
I loved every minute

"""}

result = requests.post(url, json=data).json()
print(f"Predicted rating: {min(float(result['Prediction']), 10):.2f}")
