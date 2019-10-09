#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt #调用环境


# In[24]:


plot_x = np.linspace(-1, 6, 141) #生成（-1，6）中141个点 连接成线 x即为输入变量
plot_x #显示141个点（等分）


# In[25]:


plot_Y = (plot_x - 2.5) ** 2 - 1 #损失函数y=（x-2.5）^2-1


# In[26]:


plt.plot(plot_x, plot_Y) #绘制以（x,Y）坐标的图形
plt.show() #显示图形


# In[27]:


def dJ(theta): #定义输入为theta的J函数的导函数DJ
    return 2 * (theta - 2.5) #导函数公式dJ=2（theta-2.5）


# In[28]:


def J(theta): #定义输入为theta的J函数
    return (theta - 2.5) ** 2 - 1 #函数公式J=（theta-2.5）^2-1


# theta = 0.0 #theta从0.0开始
# while True:  #循环
#     gradient = dJ(theta) #将DJ（theta）的值赋予变量gradient，即算出梯度
#     theta = theta - eta * gradient #theta向着梯度的反方向移动，“步长”为0.1
# 这里已经完成了梯度下降的过程，但是我们如何“跳出”循环？
# 即：如何判断梯度已经下降到了足够小，从而使得我们可以将该值近似看作最小值的程度？
# 尤其是，计算机对于浮点数的计算本身就携带着近似的因素，我们可以提出自己的精度要求
# 答案是：
# 当损失函数一次梯度下降过程变化的值足够小，则可以结束循环 

# In[29]:


eta = 0.1
epsilon = 1e-8 #设立阈值 当一次梯度下降损失函数变化值小于阈值则结束循环

theta = 0.0
while True:
    gradient = dJ(theta)
    last_theta = theta #保留上一次梯度下降后的theta值，用于计算下次梯度下降损失函数值的变化
    theta = theta - eta * gradient
    
    if(abs(J(theta) - J(last_theta)) < epsilon): #结束循环的判断语句，即该次梯度下降损失函数变化值小于阈值
        break #结束循环


# In[30]:


print(theta)
print(J(theta))


# 综上：
# 使得损失函数最小的输入变量是theta
# 使得损失函数最小的输出变量是J(theta)
