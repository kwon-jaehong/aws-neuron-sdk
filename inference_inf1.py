import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import torch.nn.functional as F
import torch_neuron

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴

batch_size_list = [16,32,64,128,256,512,1024,2048,4096]
mnist_test = dsets.MNIST(root='MNIST_data/',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        # (입력 뉴런, 출력 뉴런)
        self.fc1 = nn.Linear(3136, 1000)    # 7 * 7 * 64 = 3136
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = CNN()
model.eval()
model_neuron = torch.jit.load('inf1_mnist.pt')


for batch_size in batch_size_list:
    test_loader = DataLoader(dataset=mnist_test,batch_size = batch_size, shuffle = True)

    correct = 0
    start = time.time()
    for data, target in test_loader:
        output = model(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()
    end = time.time()
    # print('Test set Accuracy : {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
    print(f"batch_size : {batch_size}, {end - start:.5f} sec")
