from fastapi import FastAPI
from api.routers.inference_router import router

app = FastAPI()
app.include_router(router)