import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from dataset import DecisionDataset
from predict import Predictor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/val.jsonl');ap.add_argument('--checkpoint',default='checkpoints/system1')
    args=ap.parse_args();pred=Predictor(args.checkpoint);rows=DecisionDataset(args.data)
    # Base model produces softmax probabilities; recover logits up to an additive constant.
    examples=[]
    for r in rows:
        result=pred.predict(r['context'],r['question'],r['options'])
        probs=np.array([x['probability'] for x in result['probabilities']])
        examples.append((np.log(np.clip(probs,1e-12,1)),r['label']))
    def nll(log_t):
        t=np.exp(log_t);loss=0
        for logits,y in examples:
            z=logits/t;z=z-np.max(z);loss+=np.log(np.exp(z).sum())-z[y]
        return loss/len(examples)
    fit=minimize_scalar(nll,bounds=(np.log(.1),np.log(10)),method='bounded')
    t=float(np.exp(fit.x));out=Path(args.checkpoint)/'calibration.json'
    out.write_text(json.dumps({'temperature':t,'validation_nll':float(fit.fun)},indent=2))
    print('Temperature:',t,'saved to',out)
if __name__=='__main__':main()
