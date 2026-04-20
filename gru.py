import util
import numpy as np
import pandas as pd
from math import sqrt
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def main(df, url_output, looping, train_persentace=0.9, hidden_layer_system=3, drop_rate=0.2, epoch_system=100):

  format_file = url_output+"/[Training Percentage: {train_persentace}; Hidden Layer:{hidden_layer_system}; Drop Rate:{drop_rate}; Number of Epoch:{epoch_system}]"

  print("Feature: \nTrain Percentage:", train_persentace, "\nHidden Layer:", hidden_layer_system, "\nDrop rate:", drop_rate, "\nNumber of Epoch:", epoch_system)

  looping += 1

  plt.plot(df['Price'])
  plt.show()

  returns = df['Price']
  min_max_scaler = MinMaxScaler()

  condition=[]
  for i in range(len(df["Price"])):
    if i==0 or df["Price"][i]-df["Price"][i-1]==0:
      condition.append(0.00002)
    elif df["Price"][i]-df["Price"][i-1]>0:
      condition.append(0.00003)
    elif df["Price"][i]-df["Price"][i-1]<0:
      condition.append(0.00001)

  condition = pd.DataFrame(condition, columns = ["condition"])
  condition = condition['condition'].values.reshape(-1,1)
  condition = min_max_scaler.fit_transform(condition)

  plt.plot(returns)
  plt.show()

  returns.hist()
  plt.show()

  npa = returns.values[1:].reshape(-1, 1)
  print("NPA", len(npa), npa)
  scale = MinMaxScaler(feature_range=(0,1))
  npa = scale.fit_transform(npa)

  samples = 10
  X = []
  Y = []
  for i in range(npa.shape[0]-samples):
      X.append(npa[i:i+samples])
      Y.append(npa[i+samples][0])
      
  print("Training Data:\n\tLength is ", len(X[0:1][0]), ": \n\t\t", X[0:1])
  print("Testing Data:\n\tLength is ", len(Y[0:1]), ": \n\t\t", Y[0:1])

  X = np.array(X)
  Y = np.array(Y)

  print("Dimensions of X", X.shape, "Dimensions of Y", Y.shape)

  threshold = round(train_persentace*X.shape[0])
  trainX, trainY = X[:threshold], Y[:threshold]
  testX, testY = X[threshold:], Y[threshold:]
  print("Threshold is", threshold)

  model = keras.Sequential()

  model.add(layers.GRU(hidden_layer_system,
                      activation = "tanh",
                      recurrent_activation = "sigmoid",
                      input_shape=(X.shape[1], X.shape[2])))

  model.add(layers.Dropout(drop_rate))

  model.add(layers.Dense(1))

  model.compile(loss="mean_squared_error", optimizer = "adam")

  model.summary()

  print(X[:threshold])

  print("================================Training===================================")
  history = model.fit(trainX,
                      trainY,
                      shuffle = False,
                      epochs=epoch_system,
                      batch_size=32,
                      validation_split=0.2,
                      verbose=0)

  print("=============================Done Training=================================")

  plt.plot(history.history['loss'], label ="Training Loss")
  plt.plot(history.history['val_loss'], label ="Validation loss")
  plt.legend()
  plt.savefig(format_file.format(train_persentace=train_persentace, hidden_layer_system=hidden_layer_system, drop_rate=drop_rate, epoch_system=epoch_system)+" Validation Loss "+str(looping)+".png", dpi=500)
  plt.show()

  print(threshold)
  true_Y = testY
  pred_Y = []
  print("Number of Forecast to do ", Y.shape[0] - round(testY.shape[0]*train_persentace))

  print("================================Predicting===================================")
  pred_Y = model.predict(testX)
  print("=============================Done Predicting=================================")
  real_Y = []
  for i in testY:
    real_Y.append(i)

  array_pred = np.array(pred_Y)
  array_true = np.array(true_Y)
  
  array_pred_new = util.denormalisasi(array_pred, df["Price"].min(), df["Price"].max())
  array_true_new = util.denormalisasi(array_true, df["Price"].min(), df["Price"].max())

  MAE = mean_absolute_error(array_true_new, array_pred_new)
  RMSE = sqrt(mean_squared_error(array_true_new, array_pred_new))
  print("Value of MAE: ", MAE)
  print("Value of RMSE: ", RMSE)

  plt.plot(array_true_new, label="True Value")
  plt.plot(array_pred_new, label="Forecasted Value")
  plt.legend()
  plt.savefig(format_file.format(train_persentace=train_persentace, hidden_layer_system=hidden_layer_system, drop_rate=drop_rate, epoch_system=epoch_system)+" Forecast after denormalization "+str(looping)+".png", dpi=500)
  plt.show()

  new_testX = np.array(testX)
  new_pred_Y = []

  forecast_interval = 5
  for i in range(forecast_interval):
    print("Progress: ", i+1, "/", forecast_interval)
    if (len(new_pred_Y) == 0):
      p = model.predict(new_testX[-1].reshape(1, X.shape[1], 1))[0,0]
      print("Prdict using New_test:\n",new_testX[-(1)], p)
      new_pred_Y.append(p)
      p = np.array(p)
      a = np.append(new_testX[-1][-(samples-1):], p)
      a = a.reshape(1, X.shape[1], 1)
    else:
      p = model.predict(a)[0,0]
      print("Prdict using A:\n", a, p)
      new_pred_Y.append(p)
      p = np.array(p)
      a = np.append(a[-1][-(samples-1):], p)
      a = a.reshape(1, X.shape[1], 1)


  new_pred_Y = util.denormalisasi(np.array(new_pred_Y), df["Price"].min(), df["Price"].max())
  new_pred_Y = np.append(array_pred_new[-20:], new_pred_Y)
  plt.plot(array_true_new[-20:], label="True Value")
  plt.plot(new_pred_Y, label="New Forecasted Value")
  plt.plot(array_pred_new[-20:], label="Forecasted Value")
  plt.legend()
  plt.savefig(format_file.format(train_persentace=train_persentace, hidden_layer_system=hidden_layer_system, drop_rate=drop_rate, epoch_system=epoch_system)+" Forecast for the next 5 days "+str(looping)+".png", dpi=500)
  plt.show()



  f = open(format_file.format(train_persentace=train_persentace, hidden_layer_system=hidden_layer_system, drop_rate=drop_rate, epoch_system=epoch_system)+" Error dan Forecasted 5 days "+str(looping)+".txt", "w")
  f.write("Value of MAE:\n"+str((MAE)))
  f.write("\nValue of RMSE:\n"+str(RMSE))
  # f.write("Value of MASE:\n"+MASE)
  f.write("\n\nForecasted Value :\n"+','.join(str(x) for x in new_pred_Y))
  f.close()

  return MAE, looping