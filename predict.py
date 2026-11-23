import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from model import DynamicDecisionModel, collate

class Predictor:
    def __init__(self, checkpoint='checkpoints/system1',device=None):
        p=Path(checkpoint)
        cfg=json.loads((p/'config.json').read_text())
        self.max_length=cfg['max_length']
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer=AutoTokenizer.from_pretrained(p/'tokenizer')
        self.model=DynamicDecisionModel(cfg['base_model']).to(self.device)
        self.model.load_state_dict(torch.load(p/'model.pt',map_location=self.device,weights_only=True))
        self.model.eval()

    @torch.inference_mode()
    def predict(self,context,question,options,temperature=1.0):
        if len(options)<2 or len(set(options))!=len(options):
            raise ValueError('Provide at least two distinct options')
        if not 0<temperature<=10:raise ValueError('temperature must be in (0, 10]')
        row={'context':context,'question':question,'options':options,'label':0}
        batch=collate([row],self.tokenizer,self.max_length)
        logits=self.model(batch['input_ids'].to(self.device),batch['attention_mask'].to(self.device),batch['option_mask'].to(self.device))[0]
        probs=torch.softmax(logits/temperature,dim=-1).cpu().tolist()
        idx=max(range(len(probs)),key=lambda i:probs[i])
        return {'choice':options[idx],'index':idx,'probabilities':[{'option':v,'probability':round(prob,6)} for v,prob in zip(options,probs)]}
