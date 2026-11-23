import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class DecisionDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for row in self.rows:
            questions = row.get('questions', [row])
            if not questions or any(len(q['options']) < 2 or not 0 <= q['label'] < len(q['options']) for q in questions):
                raise ValueError('Every question needs >=2 options and a valid label')
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def collate(rows, tokenizer, max_length=128):
    contexts, questions, options, owners, qindex, labels = [], [], [], [], [], []
    for i,row in enumerate(rows):
        contexts.append(row['context'])
        for question in row.get('questions', [row]):
            j = len(questions)
            questions.append(question['question'])
            owners.append(i)
            labels.append(question['label'])
            for option in question['options']:
                options.append(option)
                qindex.append(j)
    def tokenize(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    c,q,o = tokenize(contexts),tokenize(questions),tokenize(options)
    return {'context_ids':c['input_ids'], 'context_attention':c['attention_mask'],
            'question_ids':q['input_ids'], 'question_attention':q['attention_mask'],
            'option_ids':o['input_ids'], 'option_attention':o['attention_mask'],
            'question_owner':torch.tensor(owners), 'option_question':torch.tensor(qindex),
            'labels':torch.tensor(labels)}

def to_device(batch,device): return {k:v.to(device) for k,v in batch.items()}
