import json
from pathlib import Path
from torch.utils.data import Dataset

class DecisionDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for i,row in enumerate(self.rows):
            if not isinstance(row.get('context'),str) or not isinstance(row.get('question'),str):
                raise ValueError(f'Invalid text at row {i}')
            if not isinstance(row.get('options'),list) or len(row['options'])<2 or not all(isinstance(x,str) and x.strip() for x in row['options']):
                raise ValueError(f'Invalid options at row {i}')
            if type(row.get('label')) is not int or not 0<=row['label']<len(row['options']):
                raise ValueError(f'Invalid label at row {i}')
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]
