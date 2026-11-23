import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import SystemOneV3, BASE_MODEL
from dataset import DecisionDataset, collate, to_device

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default='data/train.jsonl');ap.add_argument('--val',default='data/val.jsonl')
    ap.add_argument('--output',default='checkpoints/system1-v3');ap.add_argument('--base-model',default=BASE_MODEL)
    ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--max-length',type=int,default=128)
    ap.add_argument('--freeze-encoder',action='store_true');ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(args.base_model)
    model=SystemOneV3(args.base_model).to(device)
    if args.freeze_encoder:
        for p in model.encoder.parameters():p.requires_grad_(False)
        # Frozen encoder should remain in eval mode (no dropout).
        original_train=model.train
        def train_mode(mode=True):
            original_train(mode)
            model.encoder.eval()
            return model
        model.train=train_mode
    def loader(path,shuffle):
        return DataLoader(DecisionDataset(path),batch_size=args.batch_size,shuffle=shuffle,
                          collate_fn=lambda r:collate(r,tok,args.max_length))
    tr,va=loader(args.train,True),loader(args.val,False)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    best=float('inf')
    for epoch in range(args.epochs):
        model.train();train_total=0;train_n=0
        for raw in tr:
            batch=to_device(raw,device)
            optimizer.zero_grad(set_to_none=True)
            logits,_=model(batch)
            loss=torch.nn.functional.cross_entropy(logits,batch['labels'])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            train_total+=loss.item()*len(batch['labels']);train_n+=len(batch['labels'])
        model.eval();val_total=0;val_n=0;correct=0
        with torch.inference_mode():
            for raw in va:
                batch=to_device(raw,device);logits,_=model(batch)
                val_total+=torch.nn.functional.cross_entropy(logits,batch['labels'],reduction='sum').item()
                correct+=(logits.argmax(-1)==batch['labels']).sum().item();val_n+=len(batch['labels'])
        val_loss=val_total/val_n
        print(json.dumps({'epoch':epoch+1,'train_loss':train_total/train_n,'val_loss':val_loss,'val_accuracy':correct/val_n}),flush=True)
        if val_loss<best:
            best=val_loss
            torch.save(model.state_dict(),output/'model.pt')
            tok.save_pretrained(output/'tokenizer')
            (output/'config.json').write_text(json.dumps({'base_model':args.base_model,'max_length':args.max_length,'layers':2,'heads':4},indent=2))
    print('Best checkpoint saved:',output)
if __name__=='__main__':main()
