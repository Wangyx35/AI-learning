# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import preprocession



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

# %%定义损失函数和优化器 loss和optimizer

from torch import optim
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#%%训练网络

# 所有的网络的训练流程都是类似的，不断的执行如下的流程
# 1.输入数据
# 2.前向传播+反向传播
# 3.更新参数


for  epoch in range(2):

    running_loss = 0.0
    for i,data in enumerate(trainloader,0):

        #输入数据
        inputs, labels =data
        inputs, labels = Variable(inputs),Variable(labels)

        #梯度清零
        optimizer.zero_grad()

        #forward+backward
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()

        #更新参数
        optimizer.step()
        
        #打印log信息
        running_loss +=loss.item()
        if i%2000 == 1999 : #每2000个batch打印一次训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
    print ('Finished Training')

#%%将测试图片输入网络，计算他的label，然后与实际的label进行比较

dataiter = iter(testloader)
images, labels = dataiter.next() #一个batch返回4张图片
print('实际的label： ', ' '.join(\'%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 -0.5)).resize((400,100))


    #%%计算图片在每个类别上的分数
    outputs = net (Variable(images))
    #得分最高的那个类
    _, predicted =t.max(outputs.data, 1)

    print('预测结果： ', ' '.join('%5s'\% classes[predicted[j]] for j in range(4)))

#%%整个测试集的效果
correct = 0 #预测正确的图片数
total = 0   #总共的图片数

for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = t.max(outputs.data, 1)
    total+= labels.size()
    correct+=(predicted == labels).sum()

print('10000张册书籍中的准确率为： %d %%' %(100* correct /total))


