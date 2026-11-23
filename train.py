import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import DynamicDecisionModel, BASE_MODEL, collate
from dataset import DecisionDataset

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default='data/train.jsonl'); ap.add_argument('--val',default='data/val.jsonl')
    ap.add_argument('--output',default='checkpoints/system1'); ap.add_argument('--base-model',default=BASE_MODEL)
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--lr',type=float,default=2e-5); ap.add_argument('--max-length',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--freeze-encoder',action='store_true',help='Train only scorer (fast CPU sanity test)')
    ap.add_argument('--limit-train',type=int,default=0)
    args=ap.parse_args(); seed_everything(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(args.base_model)
    model=DynamicDecisionModel(args.base_model).to(device)
    if args.freeze_encoder:
        for param in model.encoder.parameters(): param.requires_grad_(False)
    def loader(path,shuffle):
        dataset=DecisionDataset(path)
        if shuffle and args.limit_train: dataset=torch.utils.data.Subset(dataset,range(min(len(dataset),args.limit_train)))
        return DataLoader(dataset,batch_size=args.batch_size,shuffle=shuffle,
            collate_fn=lambda rows:collate(rows,tok,args.max_length))
    tr,va=loader(args.train,True),loader(args.val,False)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=.01)
    print('Training samples:',len(tr.dataset),'validation samples:',len(va.dataset),'device:',device,flush=True)
    best=float('inf'); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); total=0
        for batch in tr:
            batch={k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch['input_ids'],batch['attention_mask'],batch['option_mask'])
            loss=torch.nn.functional.cross_entropy(logits,batch['labels'])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();total+=loss.item()
        model.eval(); losses=[];correct=0;count=0
        with torch.inference_mode():
            for batch in va:
                batch={k:v.to(device) for k,v in batch.items()}
                logits=model(batch['input_ids'],batch['attention_mask'],batch['option_mask'])
                losses.append(torch.nn.functional.cross_entropy(logits,batch['labels']).item())
                correct+=(logits.argmax(-1)==batch['labels']).sum().item();count+=len(batch['labels'])
        v=sum(losses)/len(losses)
        print(json.dumps({'epoch':epoch+1,'train_loss':total/len(tr),'val_loss':v,'val_accuracy':correct/count}))
        if v<best:
            best=v;torch.save(model.state_dict(),out/'model.pt');tok.save_pretrained(out/'tokenizer')
            (out/'config.json').write_text(json.dumps({'base_model':args.base_model,'max_length':args.max_length},indent=2))
    print('Saved best checkpoint to',out)
if __name__=='__main__':main()
