import scipy
import scipy.fftpack
import pylab
import numpy as np
from scipy import pi
import matplotlib.pyplot as plt

x = np.linspace(0,100,200)
y = np.sin(x)

y += np.random.normal(size = x.shape) * 0.1



fourier = np.fft.fft(y)
freq= np.fft.fftfreq(len(y))

idx = np.argmax(fourier)
print(freq[idx])

plt.figure()
plt.plot(x,y)

plt.figure()
plt.plot(fourier)

plt.show()
