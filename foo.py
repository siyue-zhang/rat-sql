from transformers import BertModel

BertModel.from_pretrained('bert-large-uncased-whole-word-masking')

import torch

print(torch.cuda.is_available())

print(torch.cuda.device_count())