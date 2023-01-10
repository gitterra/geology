import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Worker:
    def __init__(self, fname):
        '''
        Инициализация
        '''
        self.df = pd.read_csv(fname, decimal=',')
    
    def info(self):
        '''
        Вывод информации о датафрейма
        '''
        display(self.df.head())
        print(f'\nРазмер: {self.df.shape}')

    def get_y_collektors(self):
        '''
        Получение y_data для столбца "Коллекторы"
        '''
        # Преобразование в OHE
        enc = OneHotEncoder()
        y_data = enc.fit_transform(
            self.df['Коллекторы'].values.reshape(-1, 1)
            ).toarray().astype(np.int16)
        print(f'Размер: {y_data.shape}')
        return y_data

    def get_x_data(self, columns):
        '''
        Получение x_data
        - columns - список столбцов вида ['GGKP_korr', 'GK_korr', 'DTP_korr']
        '''
        get_x_data = self.df[columns].values.astype(np.float32)
        print(f'Размер: {get_x_data.shape}')
        return get_x_data

class Visualizer:
  def __init__(self):
    pass

  def ConfusionMatrixShow(y_true, 
                        y_pred,
                        cm_round=3,
                        figsize=(8,8),
                        title='',
                        class_labels=[80, 4, 2]):
    cm = confusion_matrix(np.argmax(y_true, axis=-1),
                          np.argmax(y_pred, axis=-1))
    # Округление значений матрицы ошибок
    cm = np.around(cm, cm_round)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Нейросеть {title}: матрица ошибок нормализованная', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    plt.gca().images[-1].colorbar.remove()  # Стирание ненужной цветовой шкалы
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    fig.autofmt_xdate(rotation=45)          # Наклон меток горизонтальной оси при необходимости
    plt.show()    

  def preprocesscollectors(self, data):
    res = []
    current = 0
    for d in data:
      if current != d:
        res.append([d, 1])
        current = d
      else:
        res[-1][1]+=1
    return res

  def createtablet(self,
                   df, 
                   start=-1, 
                   end=-1,
                   collector_predict=None,
                   KNEF_predict=None,
                   KPEF_predict=None
                   ):
    '''
    parameters:
    df - исходный датафрейм, по которому будет строиться Планшет
    start - стартовая строка датасета
    end - завершающая строка датасета ([start-end] - диапазон исходного датасета, на котором будет построен планшет)
    collector_predict - результат работы модели классификации коллекторов (список значение [2, 4 80])
    KNEF_predict - результат работы модели предсказания параметра KNEF
    KPEF_predict - результат работы модели предсказания параметра KPEF
    '''
    # Проверка на корректность введенных данных
    if end<=start and end!=-1:
      print(f'Параметр end({end}) не может быть меньше параметра start({start})')
    if start<0:
      start=0
    if end>= df.shape[0] or end==-1:
      end=df.shape[0]

    # Построение полотна с подстройкой размера под длину датасета

    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 12, 4]}, figsize=(12, (end-start)//24))
    plt.grid(which='both')
    # Автоматическая подсройка надписей
    fig.tight_layout(pad=5.0)
    #plt.subplots_adjust(hspace=0)

    # Интервал горизонтальных линий
    minor_ticks_y = np.arange(df['ГЛУБИНА'].values[start:end][0], df['ГЛУБИНА'].values[start:end][-1], .4)
    # Коллекторы
    pr_data = self.preprocesscollectors(df['Коллекторы'].values[start:end])
    if collector_predict is not None:
      pr_data_predict = self.preprocesscollectors(collector_predict)
    patterns = {
      80: ['o', '#ebebeb'],
      2: ['//', '#b4fcb4'],
      4: ['', 'white'],
      }
    ax[0].set_title('Коллекторы')
    col_width = 1    
    offset = df['ГЛУБИНА'].values[start]
    if collector_predict is not None:
      col_width =.5
      for i, d in enumerate(pr_data_predict):
        ax[0].barh(offset,  1, height=+d[1]/10, hatch=patterns[d[0]][0], color=patterns[d[0]][1], edgecolor='black', align='edge')
        offset += d[1]/10
    offset = df['ГЛУБИНА'].values[start]
    for i, d in enumerate(pr_data):      
      ax[0].barh(offset, col_width, height=+d[1]/10, hatch=patterns[d[0]][0], color=patterns[d[0]][1], edgecolor='black', align='edge')
      offset += d[1]/10
    ax[0].set_ylim(df['ГЛУБИНА'].values[end-1]+1, df['ГЛУБИНА'].values[start]-1)
    ax[0].set_xlim(0, 1)    
    ax[0].set_xticks([0, 1])  
    ax[0].set_xticklabels(['и', 'п'])
    ax[0].xaxis.tick_top()  
    ax[0].set_yticks(df['ГЛУБИНА'].values[start:end][::40])
    ax[0].grid(which='both') 
    ax[0].grid(which='minor', alpha=0.15)
    ax[0].grid(which='major', alpha=0.8)

    # График KPEF
    ax[1].plot(df.KPEF.values[start:end], df['ГЛУБИНА'].values[start:end], c='black', linewidth=1)
    if KNEF_predict is not None:
      ax[1].plot(KNEF_predict, df['ГЛУБИНА'].values[start:end], c='red', linewidth=1)
    ax[1].autoscale_view('tight')  
    ax[1].set_ylim(df['ГЛУБИНА'].values[end-1]+1, df['ГЛУБИНА'].values[start]-1)
    ax[1].set_xticks(np.arange(0, 0.5, .1))    
    ax[1].set_xticks(np.arange(0, 0.4, .01), minor=True)
    ax[1].set_yticks(minor_ticks_y, minor=True)
    ax[1].set_xticklabels([0, .1, .2, .3, .4])
    ax[1].set_yticks(df['ГЛУБИНА'].values[start:end][::40])
    ax[1].xaxis.tick_top()
    ax[1].grid(which='both') 
    ax[1].grid(b=True, which='minor', alpha=0.15)
    ax[1].grid(b=True, which='major', alpha=0.8)
    ax[1].set_title('KPEF')

    # График KNEF
    ax[2].plot(df.KNEF.values[start:end], df['ГЛУБИНА'].values[start:end], c='black', linewidth=2)
    if KPEF_predict is not None:
      ax[2].plot(KPEF_predict, df['ГЛУБИНА'].values[start:end], c='red', linewidth=1)
    ax[2].autoscale_view('tight')    
    ax[2].set_ylim(df['ГЛУБИНА'].values[end-1]+1, df['ГЛУБИНА'].values[start]-1)
    ax[2].set_xticks(np.arange(0, 1.3, .5))
    ax[2].set_xticks(np.arange(0, 1, .1), minor=True)
    ax[2].set_yticks(minor_ticks_y, minor=True)
    ax[2].set_xticklabels([0, .5, 1])
    ax[2].set_yticks(df['ГЛУБИНА'].values[start:end][::40])
    ax[2].xaxis.tick_top()
    ax[2].grid(which='both') 
    ax[2].grid(which='minor', alpha=0.15)
    ax[2].grid(which='major', alpha=0.8)
    ax[2].set_title('KNEF')

    plt.show()