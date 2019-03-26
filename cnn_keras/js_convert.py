
from keras.models import load_model
from keras.models import model_from_json

with open('model_xu_10.json', 'r') as f:
    json = f.read()
model = model_from_json(json)

