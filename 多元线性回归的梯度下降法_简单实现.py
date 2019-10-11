#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import matplotlib.pyplot as plt


# In[23]:


x = 2 * np.random.random(size = 100) #生成一组特征为1，数量为100的随机样本
y = x * 3. + 4. + np.random.normal(size = 100) #生成一组y=3x+4+噪声 噪声数量为100，服从标准正态分布


# In[24]:


X = x.reshape(-1, 1) #生成100行1列的一维数组


# In[25]:


X.shape #显示数组X的行列数


# In[26]:


y.shape #显示数组y的行列数


# 即X是一个100行1列的数组，而数组y中拥有100个元素

# In[27]:


plt.scatter(x, y)
plt.show()


# 以上各点就是我们生成的样本点

# 由于样本是随机生成的，如果想保证随机出来的样本具有可重复性，即在随机生成前添加随机的种子

# In[50]:


import numpy as np
import matplotlib.pyplot as plt


# In[51]:


np.random.seed(520) #设置随机的种子，生成样本可重复
x = 2 * np.random.random(size = 100)
y = x * 3. + 4. + np.random.normal(size = 100)


# In[52]:


X = x.reshape(-1, 1)


# In[53]:


plt.scatter(x, y)
plt.show()


# # 使用梯度下降法训练数据

# ![image.png](attachment:image.png)

# In[54]:


def J(theta, X_b, y): #定义损失函数，其中需要参数theta，和导入已知数据X_b和y
    try:
        return np.sum((y- X_b.dot(theta)) ** 2) / len(X_b) #损失函数表达式可见上图
    except:
        return float('inf') #为了防止溢出，检测到异常时返回损失函数（浮点数）的最大


# In[55]:


def dJ(theta, X_b, y): #定义损失函数的梯度
    res = np.empty(len(theta)) #构造右式中的向量res，维度跟数组theta长度一致，此时为空
    res[0] = np.sum(X_b.dot(theta) - y) #res中第一项可以特别处理，即先括号再求和
    for i in range(1, len(theta)): #剩下的n项可以做一个循环
        res[i] = (X_b.dot(theta) - y).dot(X_b[:, i]) #向量点乘即.dot() 后项实际上就是对X_b在维度上的挑选
    return res * 2 / len(X_b) #返回的时间记得除了向量外前面还有系数


# 对于之前学习的梯度下降过程，只需要进行一点改进即可

# In[56]:


def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon): #在之前的基础上引入参数X_b和y  
    
    theta = initial_theta
    i_iter = 0
    
    while i_iter < n_iters:
        grad = dJ(theta, X_b, y) #求解梯度的过程中需要调用参数X_b和y
        last_theta = theta
        theta = theta - eta * grad
        
        if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
            
        i_iter += 1
        
    return theta #gradient_descent会计算好theta的值，返回theta

#由于theta现在是高维向量，所以将跟踪theta变化的theta_history语句删除


# In[58]:


X_b = np.hstack([np.ones((len(x), 1)), x.reshape(-1,1)]) #向量X_b应该是n+1维向量,第一列全为1
initial_theta = np.zeros(X_b.shape[1]) #初始化theta向量为n+1维=X_b矩阵的列数
eta = 0.01
n_iters = 1e4
epsilon = 1e-8

theta = gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon) #这里用theta接收一下返回的结果


# In[59]:


theta


# theta[0]就是截距，theta[1]就是斜率
# ![image.png](attachment:image.png)
# 可见，构造函数时，截距是4，斜率是3，结果是相近的，说明梯度下降法成功训练了模型

# # 封装线性回归梯度下降算法

# In[62]:


from playML.LinearRegression import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit_gd(X, y)


# In[63]:


lin_reg.coef_


# In[64]:


lin_reg.inercept_


# 不知道是我没有进行封装还是没有调用，这里操作没有问题，但是实现上出现了问题

# # 疑问：如何用向量化来提速梯度下降法，在使用真实数据运行梯度下降法的时候应该注意哪些地方
