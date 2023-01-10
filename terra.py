from tensorflow.keras.models import load_model, model_from_json
import numpy as np
import joblib

class TerraDataset:
  COLUMNS = ['GGKP', 'GK', 'PE', 'DS', 'DTP', 'Wi', 'BK', 'BMK']

  def __init__(self, dataframe):    
    self._data = dataframe[self.__class__.COLUMNS].values

  def get_x_set(self):
    return self._data

class TerraModel:
  def __init__(self, filename):
    self.model = load_model(filename)

class TerraModelKol(TerraModel):
  def __init__(self, filename, kollector):
    super(TerraModelKol, self).__init__(filename)
    self.kollector = kollector

  def predict(self, x_set):
    pred = self.model.predict(x_set, verbose=0)
    result = np.array([int(p[0]>0.5) for p in pred])
    result[result==1]=self.kollector
    return result

class TerraModelKNPEF:
  def __init__(self, model_path):
    with open('geology/models/model_knpef.json', 'r') as outfile:
      config = outfile.read()
    self.model = model_from_json(
      config, custom_objects=None
    )
    self.model.load_weights(model_path)

  def predict(self, x_set):    
    my_scaler = joblib.load('geology/models/xScaler.pkl')
    xTrSc = my_scaler.transform(x_set.reshape(-1,x_set.shape[1]))
    xTrainSc1 = xTrSc[:,0:5] # Формируем обучающую выборку для вход 1
    xTrainSc2 = xTrSc[:,5:8] # Формируем обучающую выборку для вход 2
    pred = self.model.predict([xTrainSc1, xTrainSc2] , verbose=0)
    return pred