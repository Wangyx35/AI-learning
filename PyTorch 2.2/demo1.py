#In:
from __future__ import  print_function
import  torch as t

#%%构建5*3矩阵，分配空间不初始化
x = t.Tensor(5,3)
x

#%%使用[0，1]均匀分布随机初始化二维数组
x = t.rand(5,3)
x

#%%
print(x.size()) #查看X的形状
x.size()[0],x.size(1) #查看列的个数 两种写法等价

#%%加法的第一种写法
y = t.rand(5,3)
x+y

#%%加法的第二种写法
t.add(x,y)

#%% 假发的第三种写法
result = t.Tensor(5,3)  #预先分配空间
t.add(x,y,out=result)   #输出到result
result

#%%函数名后面带下划线或修改Tensor本身
print('first y')
print(y)

print('not change y')
y.add(x)#normal add,doesn't change y
print(y)

print('change y')
y.add_(x) #inplace add,change y
print(y)

#%%
print(x)
print(x[:,0])
print(x[:,1])


#%%
from  torch.autograd import  Variable

x=Variable(t.ones(2,2),requires_grad=True)

x

#%%
y=x.sum()
y

#%%
y.grad_fn

#%%
y.backward()

#%%
x.grad

#%%
y.backward()
x.grad

#%%
x.grad.data.zero_()
