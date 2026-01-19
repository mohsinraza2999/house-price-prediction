from fastapi import FastAPI
from src.routes.pydantic_model import ApiData
import uvicorn as uv
from src.routes.inference import Interference
import torch
from src.utils import config_utils
from pathlib import Path

app=FastAPI(title="Backend",summary="House Price Prediction",
            description="This is description",version="1.0.0")


@app.get("/", response_model=dict,status_code=200)
def health():
    return {"Heath":"the app health is good"}

@app.post("/predict",status_code=200)
def predict(data:ApiData)->dict:
    path_cfg=config_utils.paths_config()['model']
    PATH=path_cfg['path']
    NAME=path_cfg['name']
    # Creating interference object
    object=Interference(name=NAME,path=Path(PATH))
    

    # iterating data
    data_list=[]
    for _,value in data:
        data_list.append(value)

    # Suppose we have one sample with 13 features
    sample_data = torch.tensor([data_list], dtype=torch.float)


    #inference_cfg=config_utils.inference_config()['checkpoints']

    prediction = object.predict(sample_data)
    return {"price predicted":f" {prediction.item():.4f}"}


def run_route()->None:
    uv.run(app,host="0.0.0.0", port=8000)

if __name__=="__main__":
    run_route()