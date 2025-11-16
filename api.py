from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "online"}

@app.post("/train")
def train_model(params: dict):
    # call your training script here
    return {"started": True}
