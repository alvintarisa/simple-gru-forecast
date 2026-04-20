import gru
import env
import util

train_persentace = 0.65
hidden_layers = [1, 5, 10, 15, 20]
epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
url_data = env.GetInputData()
url_output = env.GetOutputLocation()

data = util.load_data(url_data)
for epoch in epochs:
    for hidden_layer in hidden_layers:
        result = []
        for i in range(10):
          error, order = gru.main(data, url_output+"/Best 2", looping=i, hidden_layer_system=hidden_layer, drop_rate=0.1, Epoch_system=epoch)
          result.append([order, error])
        print(result)