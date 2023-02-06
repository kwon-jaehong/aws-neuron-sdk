import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from time import time
import os
import numpy as np
import torch.neuron

# NEURON_RT_NUM_CORES 또는 NEURON_RT_VISIBLE_CORES에 지정된 총 NeuronCore 수까지 여러 번 로드하여 Inf1 인스턴스에서 병렬로 실행할 수 있습니다.
num_cores = 4 
os.environ['NEURON_RT_NUM_CORES'] = str(num_cores)

# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"

max_length=128
paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

# Convert example inputs to a format that is compatible with TorchScript tracing
example_inputs_paraphrase = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']

# Run torch.neuron.trace to generate a TorchScript that is optimized by AWS Neuron
model_neuron = torch.neuron.trace(model, example_inputs_paraphrase)

model_parallel = torch.neuron.DataParallel(model_neuron)

latency = []
# 1장씩 10000번 넣음
num_infers = 10000
for _ in range(num_infers):
    delta_start = time()
    paraphrase_classification_logits_neuron = model_parallel(*example_inputs_paraphrase)
    delta = time() - delta_start
    latency.append(delta)
## 처리 속도 평균
mean_latency = np.mean(latency)
## 1초 / 처리 속도 = 초당 처리 할수있는 이미지 수
throughput = 1 / mean_latency
print(f"1시퀀스\t{mean_latency}\t{throughput} 처리")
