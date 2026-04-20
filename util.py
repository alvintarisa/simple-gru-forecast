import pandas as pd
from google.colab import drive

def load_data(url):
  drive.mount('/content/drive')
  df = pd.read_csv(url)
  return df

"""When given a value code, the error increases sharply by 10x (Without code: 0.016 and code: 0.16)"""
def denormalisasi(data, min, max):
  print(data, min, max)
  data = data*(max-min)+min
  return data