from .tools import bcolors
import pandas as pd

class Loader:
  COLUMNS = ['GGKP', 'GK', 'PE', 'DS', 'DTP', 'Wi', 'BK', 'BMK']

  def __init__(self, filename):
    self._filename = filename
    self._df = None

  def load(self):
    print(f'{bcolors.OKBLUE}Загрузка файла {bcolors.BOLD}{self._filename}:{bcolors.ENDC}', end=' ')

  def view(self):
    print('Содержимое файла:')
    display(self._df.head())
    print(f'Количество записей: {self._df.shape[0]}')

  def rename_columns(self):
    current_column_names = list(self._df.columns)
    for i, c in enumerate(current_column_names):
      for or_c in self.__class__.COLUMNS:
        if or_c in c:
          if or_c=='GK' and 'GGKP' in c:
            continue
          if or_c=='PE' and 'KPEF' in c:
            continue
          current_column_names[i]=or_c
    self._df.columns = current_column_names    

  def get_dataframe(self):
    return self._df.copy()



class CsvLoader(Loader):
  def __init__(self, filename):
    super(CsvLoader, self).__init__(filename)

class XlsxLoader(Loader):
  def __init__(self, filename):
    super(XlsxLoader, self).__init__(filename)
    self.load()
    self.rename_columns()
    self.view()

  def load(self):
    super(XlsxLoader, self).load()
    self._df = pd.read_excel(self._filename, header=1, skiprows = range(2, 4))
    if 'KPEF' in self._df.columns:
      if self._df['KPEF'].values.max()>1:
        self._df['KPEF'] = self._df['KPEF'].values/100
    self._df.dropna(axis=0, inplace=True)
    print(f'{bcolors.OKGREEN}{bcolors.BOLD}Done{bcolors.ENDC}')
