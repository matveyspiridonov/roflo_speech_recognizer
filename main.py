import speech_recognition as sr
import time
import record_class
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


r = sr.Recognizer()

print("Говорит человек номер 1")
for i in range(3, 0, -1):
    print('{}...'.format(i))
    time.sleep(1)
record_class.record('record1')

print("Говорит человек номер 2")
for i in range(3, 0, -1):
    print('{}...'.format(i))
    time.sleep(1)
record_class.record('record2')

data1 = read("record1.wav")[1]
data2 = read("record2.wav")[1]
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Speeches')
ax1.plot(data1)
ax2.plot(data2)
plt.show()

print("Говорит контрольный человек!")
for i in range(3, 0, -1):
    print('{}...'.format(i))
    time.sleep(1)
record_class.record('record_final')


data_final = read("record_final.wav")[1]

bias = 30000
data1, data2, data_final = list(map(abs, [np.array(data1[bias:-bias // 3]),
                                          np.array(data2[bias:-bias // 3]),
                                          np.array(data_final[bias:-bias // 3])]))

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig.suptitle('Speeches')

ax1.plot(data1)
avg_d1 = moving_average(data1, 1000)
ax1.plot(avg_d1, c='r')

ax2.plot(data2)
avg_d2 = moving_average(data2, 1000)
ax2.plot(avg_d2, c='r')

ax3.plot(data_final, c='r')
avg_d3 = moving_average(data_final, 1000)
ax3.plot(avg_d3, c='b')

p1 = 1 - cos_dist(avg_d3, avg_d1)
p2 = 1 - cos_dist(avg_d3, avg_d2)

plt.show()

probabilities = [p1 /(p1 + p2), p2 /(p1 + p2)]

fig, ax = plt.subplots(figsize=(10,7))
ax.pie(probabilities, labels=['Human 1', 'Human 2'], colors= sns.color_palette('deep'), autopct=lambda x: "{:2.1%}".format(x/100))
ax.set_xlabel('Distribution of probabilities')
plt.show()
