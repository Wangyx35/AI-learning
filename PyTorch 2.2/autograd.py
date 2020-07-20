# %%  定义网络
import torch.nn as nn
import torch.nn.functional as  F
import torch as t
from torch.autograd import Variable


# %%
class Net(nn.Module):
    def __init__(self):
        # nn.Module 子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()

        # 卷积层‘1’表示输入的图片为单通道，'6'表示输出通道数
        # ‘5’表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)

        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 在nn.Module的子类中定义了forward函数，backward函数就会自动被实现（利用Autograd）
    # 在forward函数中可试用任何支持Variable支持的函数，也可以试用python基本语法
    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2 * 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape , '-1'表示自适应

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3()
        return x


net = Net()
print(net)

# 网络的可学习参数可以通过net.parameters()返回 net.named_parameters()可同时返回可学习的参数及名称

# %%
params = list(net.parameters())
print(len(params))
# %%
for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

# %% forward函数的输入和输出都是Variable，只有Variable才具有直接求导的功能，Tensor是没有的
# 所以在输入时，需要把Tensor封装成Variable
# input = Variable(t.randn(1, 1, 32, 32))
# out = net(input)
# out.size()
#
# #%%
# net.zero_grad()
# out.backward(Variable(t.ones(1,10)))
# # %%损失函数
# output = net(input)
# target = Variable(t.arange(0, 10))
# criterion = nn.MSELoss()
# loss = criterion(output, target)
# loss
