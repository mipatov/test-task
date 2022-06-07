import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse

parser = argparse.ArgumentParser(
    description='This program finds some intervals of ltv values, where mean count of users by dates is match target count')
parser.add_argument("-n", help="Intervals number", type=int, required=True)
parser.add_argument(
    "--target_mean", help="Target mean count of users by dates", default=10, type=int)
parser.add_argument(
    "--hist_step", help="Step of data distribution histagram", default=50, type=int)
parser.add_argument("--data_path", help="Data path",
                    default='task.csv', type=str)
parser.add_argument(
    "--delta", help="Required mean value accuracy", default=0.01, type=float)
parser.add_argument(
    "--base_step", help="Base step size for searching the interval boundary", default=30, type=int)
parser.add_argument(
    "--print_log", help="Print extrnded log info", action='store_true')
parser.add_argument(
    "--draw_graph", help="Show distribution histagram and intervals in the fig", action='store_true')
parser.add_argument("--save_graph_path",
                    help="Save distribution histagram and intervals into the file", type=str)
args = parser.parse_args()
n = args.n
target_mean_cnt = args.target_mean
hist_step = args.hist_step
data_path = args.data_path
base_step = args.base_step
delta = args.delta
print_log = args.print_log
draw_graph = args.draw_graph
save_graph_path = args.save_graph_path

# загрузка данных
data = pd.read_csv(data_path, index_col=0)

# отбросим нулевые ltv и выборосы
data = data.drop(data[data.ltv == 0].index)
data = data[(np.abs(stats.zscore(data.ltv)) < 3)]

# рассчитаем среднее по дням количество клиентов попадающих в диапозоны ltv с заданным шагом
# получим распределение среднего по дням количества клиетнов от ltv
x, y = [], []
v = 0

ltv_max = data.ltv.max()
while v < ltv_max:
    x.append(v)
    ltv_step_mean = data[(data.ltv > v) & (
        data.ltv < v+hist_step)]['install_date'].value_counts().mean()
    y.append(np.nan_to_num(ltv_step_mean))
    v += hist_step

result = []
log_mean_list = []

while len(result) < n:
    alpha = 1
    # одну границу диапазона задаем случайно, вторую будем двигать с некоторым шагом и считать среднее значение
    b1 = b2 = np.random.randint(ltv_max)

    # устанавливаем направление движения границы от ближайшего края к центру
    prev_d = d = 1 if b1 < ltv_max/2 else -1
    mean_cnt = 0

    while np.abs(target_mean_cnt-mean_cnt) > delta:
        if b1 != b2:
            d = np.sign(target_mean_cnt-mean_cnt)

        if prev_d != d:
            # если шаг становится достаточно маленьким и меняется направление движения,
            # то считаем границу неудачной, идем на следующую итерацию
            if step_b2 < 0.5:
                break
            alpha /= 2
            prev_d = d

        # шаг вычисляем обратнопропорционально плотности распределения среднего количества клиентов
        step_b2 = d*base_step*alpha/(y[int(b2//hist_step)]+0.01)
        b2 += step_b2
        b2 = round(b2, 2)

        # вычисляем среднее для нового интервала
        v1 = min(b1, b2)
        v2 = max(b1, b2)
        mean_cnt = data[(data.ltv > v1) & (data.ltv < v2)
                        ]['install_date'].value_counts().mean()

    # если выход из внутреннего цикла произошел по выполнению условия точности - сохраняем результат
    if np.abs(target_mean_cnt-mean_cnt) < delta:
        result.append((min(b1, b2), max(b1, b2)))
        log_mean_list.append(round(mean_cnt, 2))

# вывод результата
if(print_log):
    print(f'Target mean count = {target_mean_cnt}, delta = {delta}')
    print('-'*25)
    print('((left, right), mean)')
    print('-'*25)
    print(*list(zip(result, log_mean_list)), sep='\n')
    print('-'*25)
else:
    print(result)

# графики
def prepare_graph():
    v1 = [p[0] for p in result]
    w = [p[1]-p[0] for p in result]

    fig, ax = plt.subplots(figsize=(15, 5))

    plt.bar(x=x, height=y, width=hist_step, facecolor='#9de',
            edgecolor='#59b', alpha=0.8, align='edge')

    plt.bar(x=v1, height=max(y), width=w, facecolor='#f914',
            edgecolor='#333b', align='edge')
    plt.xticks(np.arange(0, max(x), step=100), rotation=45)


if(draw_graph):
    prepare_graph()
    plt.show()

if(save_graph_path):
    prepare_graph()
    plt.savefig(save_graph_path)
