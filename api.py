import os
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel,Field
from predict import Predictor

app=FastAPI(title='System 1 V3 Decision API')
class Question(BaseModel):
    question:str
    options:list[str]=Field(min_length=2)
class Request(BaseModel):
    context:str
    questions:list[Question]=Field(min_length=1)
@lru_cache(maxsize=1)
def model():return Predictor(os.environ.get('SYSTEM1_CHECKPOINT','checkpoints/system1-v3'))
@app.post('/decide')
def decide(request:Request):
    return model().predict_many(request.context,[q.model_dump() for q in request.questions])
@app.get('/health')
def health():return {'status':'ok'}
