import torch
from torch import nn
from types import SimpleNamespace
from model import SystemOneV3
from dataset import collate

class FakeEncoder(nn.Module):
    config=SimpleNamespace(hidden_size=16)
    def forward(self,input_ids,attention_mask):
        v=torch.nn.functional.one_hot(input_ids % 16,16).float()
        return SimpleNamespace(last_hidden_state=v)

class FakeTokenizer:
    def __call__(self,texts,**kwargs):
        ids=[[1]+[min(ord(ch),127) for ch in t[:15]] for t in texts]
        width=max(map(len,ids))
        return {'input_ids':torch.tensor([x+[0]*(width-len(x)) for x in ids]),
                'attention_mask':torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in ids])}

def test_multi_question_and_masks():
    rows=[{'context':'payment','questions':[{'question':'department','options':['Billing','Sales','Technical'],'label':0},
                                           {'question':'priority','options':['High','Low'],'label':1}]},
          {'context':'crash','question':'team','options':['Technical','Billing'],'label':0}]
    batch=collate(rows,FakeTokenizer())
    model=SystemOneV3(encoder=FakeEncoder(),hidden=16,layers=1,heads=4)
    logits,mask=model(batch)
    assert logits.shape==(3,3)
    assert mask.tolist()==[[True,True,True],[True,True,False],[True,True,False]]
    loss=torch.nn.functional.cross_entropy(logits,batch['labels'])
    loss.backward()
    assert model.scorer[-1].weight.grad is not None
