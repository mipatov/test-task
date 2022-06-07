import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ввод данных
pts = []
n = int(input('Enter points number : '))
print('Enter points values : x, y')

for i in range(n):
    pstr = input(f'#{i+1} : ')
    p = tuple(map(float,pstr.split(',')))
    pts.append(p)

# сортировка по возрастанию _x_
pts = sorted(pts,key=(lambda x:x[0]))

x = [p[0] for p in pts]
y = [p[1] for p in pts]


min_std = 0
min_std_idx = 0
a = b = 0

# перебор значений _с_ и подсчет для каждого _с_ значений _a_ и _b_ как средних значений точек слева и справа от _с_ 
for i in range(len(y)+1):
    left = y[:i]
    right = y[i:]
    mean_left = np.mean(left) if i>0 else 0
    mean_right = np.mean(right) if i<len(y) else 0
    std = np.sqrt(sum([pow(p - mean_left,2) for p in left]+[pow(p - mean_right,2) for p in right])/len(y))
    if i==0 or min_std > std:
        min_std = std
        min_std_idx = i
        a = mean_left
        b = mean_right

c = x[min_std_idx]

# вывод результата 
print('-'*40)
print(f'a = {a}\nb = {b}\nc = {c}')

fig, ax = plt.subplots()
sns.scatterplot(x=x,y=y,ax = ax)
a_x = [x[0]-1,c,x[-1]+1]
a_y = [a,b,b]
sns.lineplot(x=a_x,y=a_y,drawstyle='steps-post',color='red',ax = ax)
plt.show()



