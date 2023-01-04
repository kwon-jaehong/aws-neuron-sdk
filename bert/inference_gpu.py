# import tensorflow  # to workaround a protobuf version conflict issue
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from time import time

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("device :", device)

# 버트 모델 빌드
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)
model.to(device=device)

# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
max_length=128
paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"모델 파라미터 갯수 {pytorch_total_params}")


latency = []
# 1장씩 100번 넣음
num_infers = 100
for _ in range(num_infers):
    delta_start = time()
    paraphrase_classification_logits = model(**paraphrase)[0]
    delta = time() - delta_start
    latency.append(delta)
## 처리 속도 평균
mean_latency = np.mean(latency)
## 1초 / 처리 속도 = 초당 처리 할수있는 이미지 수
throughput = 1 / mean_latency
print(f"1시퀀스\t{mean_latency}\t{throughput} 처리")
