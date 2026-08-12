from api.api.endpoints import api_router
from fastapi import FastAPI


import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from api.api.middleware import RequestIDMiddleware

app = FastAPI()
app.add_middleware(RequestIDMiddleware)
app.include_router(api_router)


