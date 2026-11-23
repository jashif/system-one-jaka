import argparse,json
from pathlib import Path
from scipy.optimize import minimize_scalar
from evaluate import collect,metrics
from predict import Predictor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='checkpoints/system1-v3')
    ap.add_argument('--data',default='data/val.jsonl');a=ap.parse_args()
    predictor=Predictor(a.checkpoint)
    records,_=collect(predictor,a.data)
    result=minimize_scalar(lambda t:metrics(records,t)['log_loss'],bounds=(.2,10),method='bounded')
    temperature=float(result.x)
    (Path(a.checkpoint)/'temperature.json').write_text(json.dumps({'temperature':temperature},indent=2))
    print(json.dumps({'temperature':temperature,'before':metrics(records,1),'after':metrics(records,temperature)},indent=2))
if __name__=='__main__':main()
