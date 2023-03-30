import torch
import torch_neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from time import time
import os
import numpy as np




tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"

max_length=128
paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

## bert 모델 입력값 전처리
example_inputs_paraphrase = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']

## torch 뉴런으로 컴파일
model_neuron = torch.neuron.trace(model, example_inputs_paraphrase)

latency = []
# 1장씩 10000번 넣음
num_infers = 10000
for _ in range(num_infers):
    delta_start = time()
    paraphrase_classification_logits_neuron = model_neuron(*example_inputs_paraphrase)
    delta = time() - delta_start
    latency.append(delta)

## 처리 속도 평균
mean_latency = np.mean(latency)


throughput = 1 / mean_latency
print(f"1 시퀀스 처리 시간 : {mean_latency} \t\t 초당 {throughput}장 처리")
