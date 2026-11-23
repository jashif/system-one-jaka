import os
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import Predictor

app=FastAPI(title='System 1 Decision API',version='0.1.0')
class DecisionRequest(BaseModel):
    context:str=Field(max_length=20000)
    question:str=Field(min_length=1,max_length=2000)
    options:list[str]=Field(min_length=2,max_length=32)
    temperature:float=Field(default=1.0,gt=0,le=10)

@lru_cache(maxsize=1)
def predictor():return Predictor(os.getenv('CHECKPOINT','checkpoints/system1'))

@app.get('/health')
def health():return {'status':'ok'}

@app.post('/predict')
def predict(req:DecisionRequest):
    try:return predictor().predict(req.context,req.question,req.options,req.temperature)
    except ValueError as e:raise HTTPException(status_code=422,detail=str(e))
