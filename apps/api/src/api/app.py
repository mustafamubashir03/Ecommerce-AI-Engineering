import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.api.endpoints import api_router
from api.api.middleware import RequestIDMiddleware
from api.core.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

api = get_settings().api

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=api.cors_origins,
    allow_credentials=api.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.include_router(api_router)
