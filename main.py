import asyncio
from multiprocessing import Process
import uvicorn
from config import WORKERS_COUNT, HOST

if __name__ == '__main__':
    uvicorn.run("api:app", workers=WORKERS_COUNT, host=HOST, port=8000)