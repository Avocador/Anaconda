#!/usr/bin/env python
# coding: utf-8

# # numpy.array

# In[1]:


import numpy #加载numpy


# In[2]:


numpy.__version__ #查询numpy版本号


# In[3]:


import numpy as np #用np来代替numpy


# In[4]:


np.__version__ #运用


# # python list

# In[5]:


L = [i for i in range(10)]
L


# In[6]:


L[5]


# In[7]:


L[5] = 100
L


# In[8]:


L[5] = "Machine Learning"
L


# In[9]:


import array


# In[10]:


arr = array.array('i', [i for i in range(10)]) #'i'表示整型
arr


# In[11]:


arr[5]


# In[12]:


arr[5] = 100
arr


# In[13]:


arr[5] = "Machine Learing"
arr


# # numpy.array

# In[14]:


import numpy as np


# In[15]:


nparr = np.array([i for i in range(10)])
nparr


# In[16]:


nparr[5]


# In[17]:


nparr[5] = 100
nparr


# In[18]:


nparr[5] = "Machine Learning"


# In[19]:


nparr.dtype #查看numpy.array存储的类型


# In[20]:


nparr[5] = 5.0
nparr


# In[21]:


nparr.dtype


# numpy.array隐式地将nparr[5]的浮点型变换成了整数型

# In[22]:


nparr[3] = 3.14
nparr


# In[23]:


nparr.dtype


# 由于数据被变换成整数型，所以3.14只保留了3

# # 其他创建 numpy.array的方法

# In[24]:


np.zeros(10) #创建由10个0组成的数组


# In[25]:


np.zeros(10).dtype #检查该数据元素的类型，默认都是浮点数


# In[26]:


np.zeros(10, dtype = int) #通过设置dstype参数来控制数据类型


# In[27]:


np.zeros(10, dtype = int).dtype


# In[28]:


np.zeros((3, 5)) #创建元组构成的矩阵，三行五列


# In[29]:


np.zeros(shape = (3, 5), dtype = int) #元组函数shape通常会省略，同样可以设置dtype参数


# In[30]:


np.ones(10) #创建10个元素全是1的数组


# In[31]:


np.ones((3, 5)) #创建元组元素全为1的向量


# In[32]:


np.full((3, 5), 666) #将后面的值赋给元组内每个元素


# In[33]:


np.full(shape = (3, 5), fill_value = 666) #补齐略去的函数名


# np.zeros、np.ones默认类型全是浮点型,而np.full默认类型是整型

# In[34]:


np.full(shape = (3, 5), fill_value = 666.0) #通过赋值数据类型，改变默认数据类型


# In[35]:


np.full(fill_value = 666.0, shape = (3, 5)) #显式情况下可以更改函数位置


# # arange

# In[41]:


[i for i in range(0, 20, 2)] #range有三个参数，起始点(包含)，终止点(不包含)，步长


# In[42]:


np.arange(0, 20, 2) #对应range的形式，numpy中与之对应的是np.arange


# In[43]:


[i for i in range(0, 20, 0.2)] #range中步长不能为浮点数


# In[44]:


np.arange(0, 1, 0.2) #np.arange步长可以为浮点数


# In[45]:


np.arange(0, 10) #省略步长，则步长默认为1


# In[46]:


np.arange(10) #省略起始点，则起始点默认为0


# # linspace

# In[47]:


np.linspace(0, 20, 10) #同样是三个参数，起始点(包含)、终止点(包含)、等长截取点个数


# In[48]:


np.linspace(0, 20, 11) #以2为步长


# # random

# In[49]:


np.random.randint(0, 10) #生成一个[0,10)的随机整数


# In[50]:


np.random.randint(0, 10, 10) #设置第三个参数，则生成一个向量，元素随机取自[0,10)


# In[51]:


np.random.randint(4, 8, size = 10) #为了方便阅读，我们将第三个参数标识一下


# In[67]:


np.random.randint(4, 8, size = (3, 5)) #size也可以取一个元组，生成一个二维的随机矩阵


# In[68]:


np.random.seed(123) #随机种子，由于随机数实际上都是伪随机，所以可由seed保证随机相同


# In[70]:


np.random.random() #默认生成[0，1）之间的浮点数


# In[71]:


np.random.random(10) #默认参数为size，每个元素都在[0，1）之间


# In[72]:


np.random.random((3, 5) #传入一个元组，生成随机矩阵，每个元素都在[0，1）之间


# In[73]:


np.random.normal() #随机生成一个N(0,1)正态分布的浮点数


# In[74]:


np.random.normal(10, 100) #指定服从均值为10，方差为100的正态分布


# In[75]:


np.random.normal(0, 1, (3, 5)) #N(0，1)正态分布，元组代表生成随机矩阵


# In[78]:


get_ipython().run_line_magic('pinfo', 'np.random.normal')


# In[80]:


get_ipython().run_line_magic('pinfo', 'np.random')


# In[81]:


help(np.random.normal)


# In[ ]:




