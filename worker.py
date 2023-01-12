from google.colab import files
from .tools import bcolors, Metrics
from .loader import CsvLoader, XlsxLoader
from .terra import TerraDataset, TerraModelKol, TerraModelKNPEF, TerraModelKNPEF
from . import utils
import pathlib

class Worker:
  XLSX_EXTENSION = '.xlsx' # Расширение экселек
  CSV_EXTENSION = '.csv'   # Расширение csv-шек
  messages = {
      'unknown_extension': f'{bcolors.FAIL}Неизвестное расширение файла. Требуется {bcolors.BOLD}*.csv{bcolors.ENDC}{bcolors.FAIL} или {bcolors.BOLD}*.xlsx{bcolors.ENDC}',
      'create_x_set': f'{bcolors.OKBLUE}Создание набора данных:{bcolors.ENDC}',
      'done': f'{bcolors.OKGREEN}{bcolors.BOLD}Done{bcolors.ENDC}',
      'kollector_predict': f'{bcolors.OKBLUE}Определение коллекторов:{bcolors.ENDC}',      
      'knpef_predict': f'{bcolors.OKBLUE}Определение параметров KNEF и KPEF:{bcolors.ENDC}',      
      'visualize': f'{bcolors.OKBLUE}Построение планшета...{bcolors.ENDC}',      
      'download': f'{bcolors.OKBLUE}Подготовка результирующего файла{bcolors.ENDC}',
  }

  def __init__(self, filename):
    self.loader = None # Загрузчик данных
    self.x_set = None
    self.load_data(filename)
    self.create_sets()

    self.result = self.loader.get_dataframe().copy()
    self.model80 = TerraModelKol(
        filename='geology/models/model_for_80_kollector.h5',
        kollector=80)
    self.model4 = TerraModelKol(
        filename='geology/models/model_for_4_kollector.h5',
        kollector=4)
    self.kollector_predict()

    self.modelKPEF = TerraModelKNPEF(
        model_path='geology/models/model_kpef_w.h5'
    )
    self.modelKNEF = TerraModelKNPEF(
        model_path='geology/models/model_knef_w.h5'
    )
    self.knpef_predict()
    self.download()


  
  def load_data(self, filename):
    file_extension = pathlib.Path(filename).suffix # Получение расширение файла
    # Создание загрузчика, в зависимости от расширения
    if file_extension == self.__class__.XLSX_EXTENSION:
      self.loader = XlsxLoader(filename)
    elif file_extension == self.__class__.CSV_EXTENSION:
      self.loader = CsvLoader(filename)
    else:
      print(self.__class__.messages['unknown_extension'])

  def create_sets(self):
    print(self.__class__.messages['create_x_set'], end=' ')
    ds = TerraDataset(self.loader.get_dataframe())
    self.x_set = ds.get_x_set()
    print(self.__class__.messages['done'])

  def kollector_predict(self):
    print(self.__class__.messages['kollector_predict'], end=' ')
    pred80 =  self.model80.predict(self.x_set)
    pred4 =  self.model4.predict(self.x_set)
    pred80[pred80==0] = pred4[pred80==0]
    pred80[pred80==0]=79
    self.result['Коллекторы (модель)'] = pred80
    print(self.__class__.messages['done'])

  def knpef_predict(self)  :
    print(self.__class__.messages['knpef_predict'], end=' ')
    pred_kpef = self.modelKPEF.predict(self.x_set)
    pred_kpef = pred_kpef*2 - 1
    pred_kpef[pred_kpef<0]=0
    self.result['KPEF (модель)'] = pred_kpef
    pred_knef = self.modelKNEF.predict(self.x_set)
    pred_knef = pred_knef*2 - 1
    pred_knef[pred_knef<0]=0
    self.result['KNEF (модель)'] = pred_knef
    print(self.__class__.messages['done'])

  def view_result(self, show_accuracy=True):
    if show_accuracy:
      self.get_accuracy()
    display(self.result.head())
    print(self.__class__.messages['visualize'], end=' ')
    visualizer = utils.Visualizer()
    df_view = self.result.copy()
    if 'Коллекторы' in df_view.columns:
      df_view['Коллекторы'].replace(79, 2, inplace=True)
      df_view['Коллекторы'].replace(75, 2, inplace=True)
    df_view['Коллекторы (модель)'].replace(79, 2, inplace=True)
    visualizer.createtablet(df = df_view,                         
                            collector_predict = df_view['Коллекторы (модель)'],
                            KNEF_predict = df_view['KPEF (модель)'],
                            KPEF_predict = df_view['KNEF (модель)'])

  def download(self):
    print(self.__class__.messages['download'], end=' ')
    self.result.to_excel("result.xlsx")
    files.download(filename='result.xlsx')
    print(self.__class__.messages['done'])


  def get_accuracy(self):
    if 'Коллекторы' in self.result.columns:
      self.result['Коллекторы'].replace(2, 79, inplace=True)
      Metrics.accuracy(self.result['Коллекторы'].values, self.result['Коллекторы (модель)'].values)
    if 'KPEF' in self.result.columns:
      Metrics.tpe(self.result['KPEF'].values, self.result['KPEF (модель)'].values, ' KPEF')
    if 'KNEF' in self.result.columns:
      Metrics.tpe(self.result['KNEF'].values, self.result['KNEF (модель)'].values, ' KNEF')
