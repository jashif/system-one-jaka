import pytest
from model import collate
from dataset import DecisionDataset

class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        import torch
        return {'input_ids':torch.ones((len(texts),5),dtype=torch.long),'attention_mask':torch.ones((len(texts),5),dtype=torch.long)}

def test_variable_options():
    rows=[{'context':'a','question':'q','options':['a','b'],'label':0},{'context':'b','question':'q','options':['a','b','c'],'label':2}]
    batch=collate(rows,FakeTokenizer())
    assert batch['input_ids'].shape==(2,3,5)
    assert batch['option_mask'].tolist()==[[True,True,False],[True,True,True]]
    assert batch['labels'].tolist()==[0,2]
