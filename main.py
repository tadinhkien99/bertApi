import pickle
from flask import Flask
import pandas as pd
import numpy as np
app= Flask(__name__)

@app.route('/api')
def index():
  data = pd.read_csv('Analytics4Dataset.csv', error_bad_lines=False)
  data = data[['Complaint description']]
  embeddings_2 = np.load('embeddings.npy')
  # model2 = BERTopic(language="english")
  # model2 = torch.load('model.pth')
  model2 = pickle.load(open("model.sav", 'rb'))
  topics, probabilities = model2.fit_transform(data['Complaint description'], embeddings_2)
  result = model2.get_topic(3)
  # print(result)
  return result
