import argparse,json,time
import numpy as np
from sklearn.metrics import accuracy_score,log_loss
from dataset import DecisionDataset
from predict import Predictor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/test.jsonl');ap.add_argument('--checkpoint',default='checkpoints/system1');ap.add_argument('--temperature',type=float,default=1.0)
    args=ap.parse_args(); rows=DecisionDataset(args.data); pred=Predictor(args.checkpoint)
    y=[];guess=[];confidence=[];losses=[];briers=[];latencies=[]
    for row in rows:
        start=time.perf_counter()
        result=pred.predict(row['context'],row['question'],row['options'],args.temperature)
        latencies.append((time.perf_counter()-start)*1000)
        probs=np.array([x['probability'] for x in result['probabilities']]);label=row['label']
        y.append(label);guess.append(result['index']);confidence.append(float(probs.max()))
        losses.append(-np.log(max(float(probs[label]),1e-12)))
        target=np.zeros(len(probs));target[label]=1
        briers.append(float(np.sum((probs-target)**2)))
    conf=np.array(confidence);correct=np.array(y)==np.array(guess)
    ece=0.0
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(conf>=low)&(conf<high if high<1 else conf<=high)
        if mask.any():ece+=float(mask.mean()*abs(correct[mask].mean()-conf[mask].mean()))
    print(json.dumps({'n':len(y),'accuracy':accuracy_score(y,guess),'log_loss':float(np.mean(losses)),
        'multiclass_brier':float(np.mean(briers)),'ece_10_bins':ece,
        'mean_latency_ms':float(np.mean(latencies)),'p95_latency_ms':float(np.percentile(latencies,95))},indent=2))
if __name__=='__main__':main()
