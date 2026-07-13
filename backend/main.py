from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_memory.client import memory_shutdown, memory_startup
from agent_memory.routes import router as memory_router
from auth.routes import router as auth_router
from vllm.routes import router as vllm_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory_startup()
    yield
    await memory_shutdown()


app = FastAPI(title="LLM Backend Proxy - Local vLLM + Agent Memory", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vllm_router)
app.include_router(memory_router)
app.include_router(auth_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
