import torch
from time import time
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import sys
from collections import OrderedDict
import torch.neuron

## 참조 소스는 https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/craft/Craft.ipynb


## 크래프트 소스 코드 복사
os.system("git clone https://github.com/clovaai/CRAFT-pytorch.git")

# # 작업디렉토리 크래프트로 바꿈
os.chdir("./CRAFT-pytorch")

## aws에서 크래프트 실험할때 체크아웃을 이렇게 둠
os.system("git checkout e332dd8b718e291f51b66ff8f9ef2c98ee4474c8")

## 사전 모델 다운로드
model_file = './craft_mlt_25k.pth'
if not os.path.isfile(model_file):
    gdd.download_file_from_google_drive(file_id='1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ', dest_path=model_file)
    
## 현재 폴더를 python에서 import할수있게 경로 추가
sys.path.append(os.getcwd())

## 크래프트 모듈 불러옴
from craft import CRAFT
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

## 입력텐서 생성
img_size=800
x = torch.rand(1,3,img_size,img_size)

## 모델 생성하고 웨이트 로드
model = CRAFT()
model.load_state_dict(copyStateDict(torch.load(model_file, map_location='cpu')))
model.eval()

model_neuron = torch.neuron.trace(model, example_inputs=x)

## Export to saved model
# model_neuron.save("craft_neuron.pt")


latency = []
num_infers = 10000
for _ in range(num_infers):
    delta_start = time()
    y = model_neuron(x) # warmup
    delta = time() - delta_start
    latency.append(delta)

## 처리 속도 평균
mean_latency = np.mean(latency)

## 1초 / 처리 속도 = 초당 처리 할수있는 이미지 수
throughput = 1 / mean_latency
print(f"1장\t{mean_latency}\t{throughput} 처리")