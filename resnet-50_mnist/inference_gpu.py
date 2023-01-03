import torch
from torchvision import models
from torch.utils.data import DataLoader,dataset
from torchsummary import summary as summary_
import time

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("device :", device)


# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.to(device=device)
model.eval()
summary_(model,(3, 224, 224),batch_size=10)

batch_size_list = [16,32,64]
for batch_size in batch_size_list:
    image = torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)
    start = time.time()
    image = image.to(device=device)
    out = model(image)
    end = time.time()
    print(f"batch_size : {batch_size}, {end - start:.5f} sec")