import pickle
from flask import Flask
import pandas as pd
import numpy as np
app= Flask(__name__)

@app.route('/api')
def index():
  data = pd.read_csv('Analytics4Dataset.csv', error_bad_lines=False)
  data = data[['Complaint description']]
  embeddings = np.load('embeddings.npy')  
  model2 = pickle.load(open("model.sav", 'rb'))  
  topics, probabilities = model2.fit_transform(data['Complaint description'], embeddings)
  result = model2.get_topic(3)
  return str(result)
