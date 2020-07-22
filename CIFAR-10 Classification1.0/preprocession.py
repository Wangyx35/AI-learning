import torchvision as tv
import torchvision.transforms as tranforms
from torchvision.transforms import ToPILImage
import torch as t

show = ToPILImage()  # 可以把Tensor转化成Image，方便可视化

# 定义对数据的预处理

transform = transforms.Compose([
    transforms.ToTensor(),  # 转化为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])

# 训练集

trainset = tv.datasets.CIFAR10(root='E:/DataBase/',
                               train=True,
                               download=True,
                               transform=transform)

trainloader = t.utils.data.DataLoader(trainset,
                                      batch_size=4,
                                      shuffle=True,
                                      num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(root='E:/DataBase/',
                              train=False,
                              download=True,
                              transform=transform)
tsetloader = t.utils.data.DataLoader(testset,
                                     batch_size=4,
                                     shuffle=False,
                                     num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck')




# (data, label) = trainset[100]
# print(classes[label])
#show((data + 1) / 2).resize((100, 100))  在IN[]里面 可以输出图像
# dataiter = iter(trainloader)
# images,labels=dataiter.next() #返回四张图片及标签
# print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
# show(tv.utils.make_grid((images+1)/2)).resize((400,100))

