import argparse,json,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import DecisionDataset,collate,to_device
from predict import Predictor

def collect(predictor,path):
    ds=DecisionDataset(path)
    loader=DataLoader(ds,batch_size=1,collate_fn=lambda r:collate(r,predictor.tokenizer,predictor.max_length))
    records=[];latencies=[]
    with torch.inference_mode():
        for raw in loader:
            batch=to_device(raw,predictor.device)
            start=time.perf_counter()
            logits,mask=predictor.model(batch)
            if predictor.device.type=='cuda':torch.cuda.synchronize()
            elapsed=(time.perf_counter()-start)*1000
            for j,label in enumerate(batch['labels'].tolist()):
                n=int(mask[j].sum().item())
                records.append((logits[j,:n].float().cpu(),label))
                latencies.append(elapsed/len(batch['labels']))
    return records,latencies

def metrics(records,temperature=1.0):
    correct=[];loss=[];brier=[];conf=[]
    for logits,label in records:
        p=torch.softmax(logits/temperature,dim=0).numpy()
        pred=int(p.argmax());correct.append(pred==label)
        loss.append(-np.log(max(float(p[label]),1e-12)))
        onehot=np.zeros(len(p));onehot[label]=1
        brier.append(float(np.square(p-onehot).sum()))
        conf.append(float(p.max()))
    ece=0.0
    for lower in np.arange(0,1,.1):
        idx=[i for i,c in enumerate(conf) if lower<=c<(lower+.1 if lower<.9 else 1.000001)]
        if idx:ece+=len(idx)/len(conf)*abs(np.mean([correct[i] for i in idx])-np.mean([conf[i] for i in idx]))
    return {'n':len(records),'accuracy':float(np.mean(correct)),'log_loss':float(np.mean(loss)),
            'multiclass_brier':float(np.mean(brier)),'ece_10_bins':float(ece)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='checkpoints/system1-v3')
    ap.add_argument('--data',default='data/test.jsonl');a=ap.parse_args()
    predictor=Predictor(a.checkpoint);records,latencies=collect(predictor,a.data)
    result=metrics(records,predictor.temperature)
    result.update({'mean_latency_ms':float(np.mean(latencies)),'p95_latency_ms':float(np.percentile(latencies,95))})
    print(json.dumps(result,indent=2))
