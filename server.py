import os
import subprocess
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

import json

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://0.0.0.0:8000"
    "http://0.0.0.0"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RETRY_TIMEOUT = 1000


@app.get("/v1/complete/stream/",
         summary="Contact model and start streaming text from model",
         description="stream models response to prompt; tokenize prompt feed it forward to neural network ",)
async def run(request : Request):
    
    payload = json.loads(await request.body())
    
    async def inference():

        model_path = os.path.join("./", payload["model.bin"])
        command = ["./run", model_path, str(payload["temperature"]), str(payload["tokens"]), payload["input"]]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE)
        try:    
            for line in proc.stdout:
                if await request.is_disconnected():
                    break
                yield {
                        "event": "message",
                        "id": "message_id",
                        "retry": RETRY_TIMEOUT,
                        "data": line.decode('utf-8').strip()
                        }
            yield {
                 "event": "message",
                        "id": "message_id",
                        "retry": RETRY_TIMEOUT,
                        "data": "<end>"
                }
        except asyncio.CancelledError as error:
            raise error
            
        proc.wait()
      

    generate = inference()
    return EventSourceResponse(generate)


if __name__ == "__main__":
    """
    curl http://localhost:8000/v1/complete/stream/ \
    -X GET \
    -d '{"model.bin" : "stories15M.bin", "temperature" : 1.0, "tokens" : 100, "input" : "Hello world"}' \
    -H 'Content-Type: text/event-stream'
    """
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)