import argparse,json
from predict import Predictor
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='checkpoints/system1-v3')
p.add_argument('--context',required=True)
p.add_argument('--question',required=True)
p.add_argument('--options',nargs='+',required=True)
p.add_argument('--temperature',type=float)
a=p.parse_args()
print(json.dumps(Predictor(a.checkpoint).predict(a.context,a.question,a.options,a.temperature),indent=2))
