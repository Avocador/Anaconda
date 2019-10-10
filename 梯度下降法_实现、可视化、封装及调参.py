#!/usr/bin/env python
# coding: utf-8

# In[177]:


import numpy as np
import matplotlib.pyplot as plt


# In[178]:


plot_x = np.linspace(-5, 4, 251)
plot_x


# In[179]:


plot_Y = (plot_x + 3) ** 2 -4


# In[180]:


plt.plot(plot_x, plot_Y)
plt.show()


# In[181]:


def dJ(theta):
    return 2 * (theta + 3)


# In[182]:


def J(theta):
    return (theta + 3) ** 2 - 4


# In[86]:


eta = 0.1
epsilon = 1e-8
theta = -4.0
while True:
    grad = dJ(theta)
    last_theta = theta
    theta = theta - eta * grad
    if abs(J(theta) - J(last_theta)) < epsilon:
        break
print(theta)
print(J(theta))


# 以上即为（给定损失函数的）梯度下降法
# 需设置三个参数 步长eta 小量epsilon 起始点theta
# 下面进行梯度下降可视化处理

# In[87]:


eta = 0.1
epsilon = 1e-8
theta = -4.0
theta_history = [theta] #用数组theta_history记录下第一个theta值
while True:
    last_theta = theta
    theta = theta - eta * grad
    grad = dJ(theta)
    theta_history.append(theta) #将theta沿梯度变化的值依次添加到数组theta_history中
    if abs(J(theta) - J(last_theta)) < epsilon:
        break
print(theta)
print(J(theta))


# In[88]:


plt.plot(plot_x, plot_Y) #绘制坐标为（x,J（x））的图像
plt.plot(np.array(theta_history), J(np.array(theta_history)), color = "r", marker = "+") #绘制坐标为theta沿梯度下降的值，颜色为红，形状为+
plt.show() #显示绘制的图形


# 开始时下降比较快，因为梯度比较大，随着梯度平缓，下降速度放慢直到小于epsilon，结束搜索跳出循环

# In[89]:


len(theta_history)


# 可以看出，经过42次的梯度下降，即theta值更新了42次，我们近似得到了局部最小值

# # 封装
# 为了可以更好地更替该模型的三个参数：
# 即eta, epsilon, theta. 我们对以上的代码进行封装。
# 用户只需要传来三个参数即可调用梯度下降法的模型

# In[120]:


def gradient_descent(initial_theta, eta, epsilon): #定义封装函数gradient_descent，参数即为theta,eta,epsilon
    theta = initial_theta #初始的theta值即为输入的theta值
    theta_history.append(initial_theta) #将输入的theta值即初始的theta值添加进数组theta_history
    
    while True:
        grad = dJ(theta)
        last_theta = theta
        theta = theta - eta * grad
        theta_history.append(theta)
        
        if abs(J(theta) - J(last_theta)) < epsilon:
            break
        
def plot_theta_history(): #将绘制图形函数进行封装
    plt.plot(plot_x,plot_Y)
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color = "r", marker = "+")
    plt.show()


# In[121]:


eta = 0.1 #输入eta的值
epsilon = 1e-8 #输入epsilon的值
theta_history = [] #初始的数组theta_history为空，必须在这里设立，放在封装函数中会导致此次结果覆盖下一次的结果，即theta_history不更新
gradient_descent(-4.0, eta, epsilon) #输入函数的各个参数
plot_theta_history() #调用封装好的绘图函数


# In[122]:


len(theta_history)


# # 调参

# 尝试在不同参数的情况下，梯度下降法的效果.以下尝试仅变动eta值：

# In[123]:


eta = 0.01
epsilon = 1e-8
theta_history = [] #初始的数组theta_history为空，必须要有
gradient_descent(-4, eta, epsilon)
plot_theta_history()


# In[124]:


len(theta_history)


# 尝试在不同参数的情况下，梯度下降法的效果.以下尝试仅变动initial_theta值：

# In[125]:


eta = 0.1
epsilon = 1e-8
theta_history = []
gradient_descent(4.0, eta, epsilon)
plot_theta_history()


# In[126]:


len(theta_history)


# 当eta很大时，会导致梯度下降法的精度降低：

# In[132]:


eta = 1.0
epsilon = 1e-8
theta_history = []
gradient_descent(-4.0, eta, epsilon)
plot_theta_history()


# In[133]:


len(theta_history)


# 当eta过大时，会导致梯度下降法报错：

# In[134]:


eta = 2.0
epsilon = 1e-8
theta_history = []
gradient_descent(-4.0, eta, epsilon)
plot_theta_history()


# 此时经过梯度下降，损失函数值反而更大了，随着这个过程的循环，损失函数值越来越大直至报错。

# 对循环进行检测

# In[135]:


def J(theta): #定义J（theta）函数
    try: #检测成功
        return (theta + 3) ** 2 - 4 #返回J（theta）的值
    except: #检测失败
        return float('inf') #返回浮点数的最大值


# In[136]:


eta = 2.0
epsilon = 1e-8
theta_history = []
gradient_descent(-4.0, eta, epsilon)


# 虽然没有抛出异常，但是该程序为死循环。
# 因为在while True的循环中，判断语句为if abs(J(theta) - J(last_theta)) < epsilon。
# 随着梯度下降的过程，J（theta）不断地增大直至无穷，“无穷-无穷”在Python里定位为not number，而不是0。
# 所以if的条件永远不会触发，所以是死循环。

# # 设立循环上限参数，避免死循环

# 引入新的参数：循环次数的上限 n_iters

# In[187]:


def gradient_descent(initial_theta, eta, n_iters, epsilon):  #引入参数n_iters来表示循环次数的上限
    theta = initial_theta
    theta_history.append(initial_theta)
    i_iter = 0 #设描述循环次数的变量i_iter
    
    while i_iter < n_iters: #当i_iter<n_iters时进行循环
        grad = dJ(theta)
        last_theta = theta
        theta = theta - eta * grad
        theta_history.append(theta)
        
        if abs(J(theta) - J(last_theta)) < epsilon:
            break
            
        i_iter += 1 #每次循环使得计数i_iter+1


# In[188]:


eta = 2.0
epsilon = 1e-8
n_iters = 5
theta_history = []
gradient_descent(-4.0, eta, n_iters, epsilon)


# In[189]:


len(theta_history)


# 由于循环次数的限制，即在6（六次更新）=1（一次输入）+5（五次循环）的限制下避免了死循环
# 注意：n_iters不可过大，否则由于循环次数过多也会报错

# In[190]:


theta_history[-1]


# 输出最后一次循环后的theta的值，即用索引[-1]来调取数组中倒数第一个值

# In[191]:


plot_theta_history()


# 可视化展示了随着梯度下降法的循环，损失函数值不断上升的过程

# eta是一个超参数，它是由梯度的取值来决定取值是否合适的，一般而言，我们将eta取0.01更加保守和适用。
