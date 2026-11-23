import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "answerdotai/ModernBERT-base"

class DynamicDecisionModel(nn.Module):
    """Jointly encode context/question/option pairs; normalize over candidate options."""
    def __init__(self, base_model=BASE_MODEL):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        h = self.encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(h // 2, 1))

    def forward(self, input_ids, attention_mask, option_mask=None):
        # [B, O, T] -> [B*O, T]
        b, o, t = input_ids.shape
        out = self.encoder(input_ids=input_ids.reshape(b*o,t), attention_mask=attention_mask.reshape(b*o,t))
        # CLS representation for ModernBERT
        pooled = out.last_hidden_state[:, 0]
        scores = self.scorer(pooled).reshape(b,o)
        if option_mask is not None:
            scores = scores.masked_fill(~option_mask.bool(), torch.finfo(scores.dtype).min)
        return scores


def format_pair(context, question, option):
    return f"Context: {context} Question: {question} Candidate answer: {option}"


def collate(batch, tokenizer, max_length=256):
    max_options = max(len(x['options']) for x in batch)
    flat = []
    masks = []
    for row in batch:
        n = len(row['options'])
        if n < 2:
            raise ValueError('Each example requires at least 2 options')
        flat.extend([format_pair(row['context'], row['question'], v) for v in row['options']])
        flat.extend([format_pair('', '', '')] * (max_options-n))
        masks.append([True]*n+[False]*(max_options-n))
    enc = tokenizer(flat, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    b = len(batch)
    return {'input_ids':enc['input_ids'].reshape(b,max_options,-1),
            'attention_mask':enc['attention_mask'].reshape(b,max_options,-1),
            'option_mask':torch.tensor(masks,dtype=torch.bool),
            'labels':torch.tensor([x['label'] for x in batch],dtype=torch.long)}
