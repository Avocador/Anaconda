#!/usr/bin/env python
# coding: utf-8

# 准备模块和数据点:

# In[1]:


import numpy as np
import pylab as pl
from scipy import interpolate
import matplotlib.pyplot as plt


# In[2]:


x = np.linspace(0, 2 * np.pi + np.pi/4, 10)
y = np.cos(x)
print(x)
print(y)


# In[3]:


plt.xlabel(u'x')
plt.ylabel(u'y')

plt.plot(x, y, "*", color = "r", label = u'data')
plt.plot(x, y, color = "b", label = u'line')

pl.legend()
pl.show()


# 用三次样条插值数据x和y：

# In[4]:


t = interpolate.splrep(x, y)
print(t)


# 显示三次样条插值拟合后的曲线(完整代码)：

# In[6]:


import numpy as np
import pylab as pl
from scipy import interpolate
import matplotlib.pyplot as plt

pl.rcParams['font.sans-serif'] = ['Kaiti']
pl.rcParams['font.size'] = 10
pl.rcParams['text.color'] = 'b'

x = np.linspace(0, 2 * np.pi + np.pi/4, 10)
y = np.cos(x)
plt.xlabel(u'x')
plt.ylabel(u'y')
plt.plot(x, y, "*", color = "r", label = u'data')
plt.plot(x, y, color = "b", label = u'line')

t = interpolate.splrep(x, y)

x0 = np.linspace(0, 2 * np.pi + np.pi/4, 1000)
y0 = interpolate.splev(x0, t)
plt.plot(x0, y0, color = "G", label = u'splrep')

pl.legend()
pl.show()


# In[ ]:




