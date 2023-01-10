from sklearn.metrics import accuracy_score

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Metrics:
  @staticmethod
  def accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f'{bcolors.OKCYAN}{bcolors.BOLD}Точность: {bcolors.ENDC} {round(acc*100, 3)}%')

  @staticmethod
  def tpe(y_true, y_pred, name=''):
    true_value = y_true
    true_value[true_value==0]=0.0001
    pred_value = y_pred
    error = abs(true_value.mean() - pred_value.mean())/true_value.mean()
    print(f'{bcolors.OKCYAN}{bcolors.BOLD}Погрешность{name}: {bcolors.ENDC} {round(error*100, 3)}%')
    return 