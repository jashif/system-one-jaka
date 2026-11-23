import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from model import SystemOneV3
from dataset import collate, to_device

class Predictor:
    def __init__(self,checkpoint='checkpoints/system1-v3',device=None):
        path=Path(checkpoint)
        config=json.loads((path/'config.json').read_text())
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer=AutoTokenizer.from_pretrained(path/'tokenizer')
        self.model=SystemOneV3(config['base_model'],layers=config['layers'],heads=config['heads']).to(self.device)
        self.model.load_state_dict(torch.load(path/'model.pt',map_location=self.device,weights_only=True))
        self.model.eval();self.max_length=config['max_length']
        temperature_path=path/'temperature.json'
        self.temperature=json.loads(temperature_path.read_text())['temperature'] if temperature_path.exists() else 1.0

    def predict_many(self,context,questions,temperature=None):
        if not questions or any(len(q['options'])<2 for q in questions):
            raise ValueError('Each question must have at least two options')
        row={'context':context,'questions':[dict(q,label=0) for q in questions]}
        batch=to_device(collate([row],self.tokenizer,self.max_length),self.device)
        with torch.inference_mode(): logits,mask=self.model(batch)
        temp=temperature if temperature is not None else self.temperature
        if temp<=0:raise ValueError('temperature must be positive')
        results=[]
        for j,q in enumerate(questions):
            n=len(q['options'])
            probabilities=torch.softmax(logits[j,:n]/temp,dim=-1).cpu().tolist()
            index=int(torch.argmax(logits[j,:n]).item())
            results.append({'question':q['question'],'choice':q['options'][index], 'index':index,
                            'probabilities':[{'option':v,'probability':round(float(p),6)} for v,p in zip(q['options'],probabilities)]})
        return {'decisions':results}

    def predict(self,context,question,options,temperature=None):
        return self.predict_many(context,[{'question':question,'options':options}],temperature)['decisions'][0]
