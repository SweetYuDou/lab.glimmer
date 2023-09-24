import torch
from torch import nn
from torch.utils.data import DataLoader
from mymodule import MyDatasets,Model
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import torchvision.transforms as transforms


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = MyDatasets('train_dataset_5.pickle',transform=None)

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)

# 创建网络模型
#model = Model()
model = models.resnet18()
model.fc = nn.Linear(512, 5)

# 损失函数
loss = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters(),lr=0.005,)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 开始循环训练
for epoch in range(30):
    model.train()
    sum_loss = 0.0
    accurate = 0
    for data in train_dataloader:
        x,y = data
        output = model(x)
        loss_in = loss(output,y)
        optimizer.zero_grad()
        loss_in.backward()
        optimizer.step()

        sum_loss += loss_in
        accurate += (output.argmax(1) == y).sum().float()

    print('第{}轮训练集的正确率:{:.2f}%,损失:{:.5f}'.format(epoch+1 , accurate/len(train_data)*100 ,sum_loss))

# 保存模型
torch.save(model,'CIFAR_CNN_model.pth')