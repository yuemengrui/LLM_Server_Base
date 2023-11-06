# *_*coding:utf-8 *_*
import uvicorn
from fastapi import FastAPI
from configs import *

app = FastAPI(title="LLM_Server_Base", docs_url=None, redoc_url=None)

if __name__ == '__main__':
    config = uvicorn.Config(app=app, host=FASTAPI_HOST, port=FASTAPI_PORT, workers=1)
    server = uvicorn.Server(config)
    import logger
    from info import app_registry

    app_registry(app)
    server.run()
