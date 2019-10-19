#!/usr/bin/env python
# coding: utf-8

# # numpy.array中的运算

# # 向量的数乘

# In[1]:


n = 10
L = [i for i in range(n)] #生成元素是0到9的十元一维向量


# In[2]:


2 * L #在python中2*L是将两个L衔接起来，并非是数乘


# In[3]:


A = []
for e in L:
    A.append(2 * e)
A #可以通过循环的方式，对每个元素进行乘法


# In[4]:


n = 1000000
L = [i for i in range(n)] #为了验证效率，生成一个特别大的向量


# In[5]:


get_ipython().run_cell_magic('time', '', 'A = []\nfor e in L:\n    A.append(2 * e) #可以看出效率非常的慢')


# In[6]:


get_ipython().run_cell_magic('time', '', 'A = [2 * e for e in L] #效率比上面高但依然慢')


# 上面是原生python中对于list的运算，下面对比numpy中向量的运算

# In[7]:


import numpy as np #搭建numpy环境
L = np.arange(n) #生成跟上面一样的向量


# In[8]:


get_ipython().run_cell_magic('time', '', 'A = np.array(2 * e for e in L) #速度比python原生的运算快了非常多')


# In[9]:


get_ipython().run_cell_magic('time', '', 'A = 2 * L #numpy完全支持常数点乘向量，速度非常快')


# In[10]:


A


# In[13]:


n = 10
L = np.arange(n)
2 * L #选取较小的向量，更直观的展示数乘的结果


# ## Universal  Functions 将数组当作向量或者矩阵运算

# In[14]:


X = np.arange(1, 16).reshape((3, 5))
X #生成一个元素是0-15的4x4的矩阵


# In[15]:


X + 1 #对于矩阵每一个元素都+1


# In[16]:


X - 1 #对于矩阵每一个元素都-1


# In[17]:


X * 2 #矩阵X数乘常数2(对于矩阵每一个元素都*2)


# In[18]:


X / 2 #浮点数的除法(对于矩阵每一个元素都/2，元素类型为浮点数)


# In[19]:


X // 2 #整数的除法(对于矩阵每一个元素都/2，元素类型为整数型)


# In[20]:


X ** 2 #对于矩阵每一个元素都乘方


# In[21]:


X % 2 #对于矩阵每一个元素对2都取余


# In[22]:


1 / X #对于矩阵每一个元素都取倒数


# 矩阵的特殊运算

# In[24]:


np.abs(X) #对于矩阵每一个元素都取绝对值


# In[25]:


np.sin(X) #对于矩阵每一个元素都取sin值


# In[26]:


np.cos(X) #对于矩阵每一个元素都取cos值


# In[27]:


np.tan(X) #对于矩阵每一个元素取tan值


# In[28]:


np.exp(X) #对于矩阵每一个元素取e的相应元素次方


# In[29]:


np.power(3, X) #对于矩阵每一个元素取3的相应元素次方


# In[30]:


3 ** X #这种写法跟上面的结果是一致的(注意和X**3的区别)


# In[31]:


np.log(X) #对于矩阵每一个元素取以e为底的自然对数


# In[32]:


np.log2(X) #对于矩阵每一个元素取以2为底的对数


# In[33]:


np.log10(X) #对于矩阵每一个元素取以10为底的对数


# ## 矩阵运算

# In[34]:


A = np.arange(4).reshape(2, 2)
A #创建一个元素是0-3的2x2矩阵


# In[35]:


B = np.full((2, 2), 10)
B #创建一个元素全为10的2x2矩阵


# In[36]:


A + B #两个矩阵相加(对应元素求和)


# In[37]:


A - B #两个矩阵相减(对应元素做差)


# In[38]:


A * B #A和B矩阵对应元素相乘


# In[39]:


A / B #A和B矩阵对应元素相除


# numpy中的运算符都是根据矩阵元素来定义的，而不是矩阵本身

# numpy中标准的矩阵之间的运算

# In[40]:


A.dot(B) #A矩阵和B矩阵的乘法(对应行乘以对应列再相加)


# In[41]:


A.T #矩阵的转置(行列互换)


# In[42]:


C = np.full((3, 3), 6) #生成一个元素都为6的3x3矩阵


# In[43]:


A + C #我们要保证这两个矩阵是可以运算的


# In[44]:


A.dot(C) #我们要保证这两个矩阵是可以运算的


# ## 向量和矩阵的运算

# In[51]:


v = np.array([1, 2]) #生成向量v


# In[52]:


A #调用元素为0-3的2x2矩阵A


# In[53]:


v + A #向量和矩阵中的每一行做加法(对应元素相加)，在数学中无定义


# In[54]:


np.vstack([v] * A.shape[0]) #将向量v按照A的行数叠成矩阵


# In[55]:


np.vstack([v] * A.shape[0]) + A #经过堆叠之后矩阵相加


# In[56]:


np.tile(v, (2, 1)) #将向量v在行方向上堆叠2次，在列方向上堆叠1次


# In[57]:


np.tile(v, (2, 1)) + A #向量v经过tile函数堆叠之后与矩阵A相加


# In[58]:


v #向量v


# In[59]:


A #矩阵A


# In[60]:


v * A #向量v和矩阵A中的每一个向量做乘法(对应元素相乘)


# In[61]:


v.dot(A) #1x2的矩阵乘以2x2的矩阵得到一个1x2的矩阵


# In[62]:


A.dot(v) #此处的向量v是被当作列向量处理，即1x2矩阵乘以2x2矩阵


# 对于一个向量和矩阵的乘法，numpy会自动帮我们判断是行向量还是列向量

# ## 矩阵的逆

# In[63]:


A #2x2的矩阵A


# In[64]:


np.linalg.inv(A) #linalg意为线性代数,inv意为逆


# In[65]:


invA = np.linalg.inv(A) #将A的逆矩阵赋值给invA


# In[66]:


A.dot(invA) #矩阵A乘以A的逆矩阵=单位矩阵(主对角线为1)


# In[67]:


invA.dot(A) #A的逆矩阵乘以矩阵A=单位矩阵


# In[68]:


X = np.arange(16).reshape((2, 8))
X #生成一个2x8的矩阵


# In[69]:


np.linalg.inv(X) #矩阵X没有逆，所以报错


# 伪逆矩阵

# In[70]:


pinvX = np.linalg.pinv(X) #pinv意为伪逆，将X的伪逆赋值给pinvX矩阵


# In[71]:


pinvX #显示X的伪逆矩阵


# In[72]:


pinvX.shape #X的伪逆矩阵是2x8的矩阵(X是8x2的矩阵)


# In[73]:


X.dot(pinvX) #X乘以X的伪逆矩阵为单位阵(主对角线外极小的数值是计算机浮点数误差造成)


# # 聚合操作（把一组值变成一个值）

# In[3]:


import numpy as np #加载numpy


# In[4]:


L = np.random.random(100) #生成100个[0，1)之间的随机数


# In[5]:


L #显示数组L


# In[6]:


sum(L) #python中可以通过sum计算list列表的和，numpy也支持sum计算数组元素的和


# In[7]:


np.sum(L) #numpy也自带数组元素求和的函数


# In[8]:


big_array = np.random.rand(1000000) #用元素数量为1000000的数组为例
get_ipython().run_line_magic('timeit', 'sum(big_array)')
get_ipython().run_line_magic('timeit', 'np.sum(big_array) #通过比较可以发现,numpy自带运算效率更高')


# In[9]:


np.min(big_array) #计算数组中最小的值


# In[10]:


np.max(big_array) #计算数组中最大的值


# In[12]:


big_array.min() #可以直接使用面向对象的调用方式,此为调用最小值


# In[13]:


big_array.max() #此为调用最大值


# In[14]:


big_array.sum() #此为调用求和值


# 两种方法都可以，但一般还是推荐使用np.函数(对象)的形式

# In[30]:


X = np.arange(16).reshape(4, -1)
X #创建一个元素为0-15的4x4的矩阵


# In[31]:


np.sum(X) #聚合运算默认计算所有元素的和


# In[32]:


np.sum(X, axis = 0) #每一列的元素求和(axis沿着行的维度，即逐列求和)


# In[33]:


np.sum(X, axis = 1) #每一行的元素求和(axis沿着列的维度，即逐行求和)


# In[34]:


np.prod(X) #对矩阵中每一个元素做乘法


# In[37]:


np.prod(X + 1)


# In[41]:


np.mean(X) #求矩阵所有元素的平均值


# In[42]:


np.median(X) #求矩阵所有元素的中位数


# In[43]:


v = np.array([1, 1, 2, 2, 10])
np.mean(v)


# In[44]:


np.median(v) #有的时候中位数比均值更能反应总体的情况


# In[45]:


np.percentile(big_array, q = 50) #在100个[0，1)的随机数中，百分之50的元素都小于显示值


# In[46]:


np.median(big_array) #百分之50位数等于中位数


# In[47]:


np.percentile(big_array, q = 100) #显示big_array百分之百位数


# In[48]:


np.max(big_array) #百分之百位数等于最大值


# In[50]:


for percent in [0, 25, 50, 75, 100]: #输出最小值、1/4中位数、中位数、3/4中位数、最大值
    print(np.percentile(big_array, q = percent))


# In[52]:


np.var(big_array) #求一组样本的方差


# In[53]:


np.std(big_array) #求一组样本的标准差


# In[57]:


x = np.random.normal(0, 1, size = 1000000) #取1000000个服从标准正态分布N(0，1)的随机数


# In[58]:


np.mean(x) #验证1000000个标准正态分布随机数的均值


# In[59]:


np.std(x) #验证1000000个标准正态分布随机数的标准差


# # arg运算(索引)

# In[61]:


x #x是1000000个服从标准正态分布N(0，1)的随机数组成的向量


# In[62]:


np.min(x) #调用向量中最小的值


# In[63]:


np.argmin(x) #返回最小值的索引


# In[64]:


x[59712] #根据索引可以找到向量中的最小值


# In[65]:


np.argmax(x) #返回最大值的索引


# In[66]:


x[19855] #调用向量x中索引为19855的值


# In[67]:


np.max(x) #验证最大值索引对应的值是否是向量中的最大值


# ## 排序和使用索引

# In[68]:


x = np.arange(16)
x #生成一个元素为0-15的向量


# In[69]:


np.random.shuffle(x) #乱序处理
x


# In[70]:


np.sort(x) #排序处理(由小到大)


# In[71]:


x #np.sort(x)只是返回一个排序好的x，没有对x进行修改


# In[72]:


x.sort() #对向量x本身进行排序的操作


# In[73]:


x #向量x已经被排序并且修改保存


# In[74]:


X = np.random.randint(10, size = (4, 4))
X #生成一个元素为[0，10)整数的4x4随机矩阵


# In[75]:


np.sort(X) #默认将矩阵X中每一行进行排序处理


# In[76]:


np.sort(X, axis = 1) #注意，此时的axis默认值是1，沿着列排序即对每一行排序


# In[77]:


np.sort(X, axis = 0) #将矩阵X中每一列进行排序处理


# In[78]:


x #生成一个元素为0-15的向量


# In[79]:


np.random.shuffle(x) #对向量x进行乱序处理


# In[80]:


x #显示乱序后的向量x


# In[81]:


np.argsort(x) #数组中存放的是乱序前元素的索引，即原先元素现在的索引


# In[84]:


np.partition(x, 3) #以3为标定点，比3小的都放在左侧，比3大的都放在右侧，不是有序的


# In[85]:


np.argpartition(x, 3) #对于索引按照3的索引为标定点区分元素值的大小


# In[86]:


X #生成一个元素为[0，10)整数的4x4随机矩阵


# In[87]:


np.argsort(X, axis = 1) #按行进行排序，返回每个值的索引


# In[88]:


np.argsort(X, axis = 0) #按列进行排序，返回每个值的索引


# In[89]:


np.argpartition(X, 2, axis = 1) #以2为标定点，按行排序，返回每个值的索引


# # Fancy Indexing

# In[90]:


import numpy as np #加载numpy


# In[91]:


x = np.arange(16)
x #生成元素为0到15的向量x


# In[92]:


x[3] #索引返回单个元素


# In[93]:


x[3:9] #以切片的方式索引返回一组元素


# In[94]:


x[3:9:2] #以切片的方式，索引步长为2，索引返回一组元素


# In[95]:


[x[3], x[5], x[8]] #手动添加没有规律的索引返回元素


# In[96]:


ind = [3, 5, 8] #生成存储了我们想要的索引的列表


# In[97]:


x[ind] #在x[]中传入写有索引的列表，则可索引返回一组元素


# In[98]:


ind = np.array([[0, 2],
                [1, 3]])
x[ind] #索引数组也可以是二维数组，则可索引返回二维矩阵(索引值的排行)


# In[99]:


X = x.reshape(4, -1)
X #将向量x转换成4x4的矩阵X


# In[100]:


row = np.array([0, 1, 2]) #将行的索引存储在向量row中
col = np.array([1, 2, 3]) #将列的索引存储在向量col中
X[row, col] #在[]中分别传入变量row和变量col


# In[101]:


X[0, col] #仅输出索引=0的行，索引为变量col的列组成的矩阵


# In[102]:


X[:2, col] #仅输出[索引=0,索引=2)的行，索引为变量col的列组成的矩阵


# 将布尔值当作索引

# In[103]:


col = [True, False, True, True] #显示索引为True的元素，不显示索引为Flase的元素


# In[104]:


X[1:3, col] #仅输出[索引=1,索引=3)的行，索引为变量col的列组成的矩阵


# # numpy.array的比较

# In[105]:


x #生成元素为0到15的向量x


# In[106]:


x < 3 #判断向量x中的所有元素是否小于3，小于为True，大于等于为False


# In[107]:


x > 3 #判断向量x中的所有元素是否大于3


# In[108]:


x <= 3 #判断向量x中的所有元素是否小于等于3


# In[109]:


x >= 3 #判断向量x中的所有元素是否大于等于3


# In[110]:


x == 3 #判断向量x中的所有元素是否等于3


# In[111]:


x != 3 #判断向量x中的所有元素是否不等于3


# In[112]:


2 * x == 24 - 4 * x #判断向量x中所有元素是否满足2x=24-4x


# In[113]:


X #调用元素为0到15的4x4矩阵X


# In[114]:


X < 6 #判断矩阵X中所有元素是否满足X>6


# In[115]:


x #假设向量x为16个样本，每个样本只有一个特征


# In[116]:


np.sum(x <= 3) #计算特征小于等于3的样本数(True=1,False=0)


# In[117]:


np.count_nonzero(x <= 3) #数出数组中有多少个非0元素(True=1,False=0)


# In[118]:


np.any(x == 0) #给定的布尔数组中如果有任意一个值是True，则为ure，否则为False


# In[119]:


np.any(x < 0) #是否有任意一个值都小于0


# In[120]:


np.all(x >= 0) #给定的布尔数组所有元素都是True，则为ure，否则为False


# In[121]:


np.all(x > 0)


# In[122]:


X #调用元素为0到15的4x4矩阵X


# In[123]:


np.sum(X % 2 == 0) #计算矩阵X中为偶数的元素的数量


# In[124]:


np.sum(X % 2 ==0, axis = 1) #沿着列的方向，即计算每一行中为偶数的元素的数量


# In[125]:


np.sum(X % 2 ==0, axis = 0) #沿着行的方向，即计算每一列中为偶数的元素的数量


# In[126]:


np.all(X > 0, axis = 1) #沿着列的方向，即判断每一行元素所有是否都大于0


# In[127]:


x #假设向量x为16个样本，每个样本只有一个特征


# In[128]:


np.sum((x > 3) & (x < 10)) #计算向量x中大于3小于10的元素数量(注意是用一个&连接)


# In[129]:


np.sum((x > 3) && (x < 10))


# In[130]:


np.sum((x % 2 == 0) | (x > 10)) #计算向量x中为偶数或大于10的元素数量(注意是用一个|连接)


# In[132]:


np.sum(~(x == 0)) #计算向量x中非等于0的元素数量(注意是用一个~连接)


# In[135]:


x[x < 5] #将布尔数组作为索引，返回所有满足条件的值组成的子数组


# In[136]:


x[x % 2 == 0] #向量x中为偶数的元素组成的子数组


# In[141]:


X[X[:, 3] % 3 == 0, :] #挑选出行数的最后一位ied特征值可以被3整除组成子矩阵


# Pandas
