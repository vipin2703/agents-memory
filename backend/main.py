from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from vllm.routes import router as vllm_router

app = FastAPI(title="LLM Backend Proxy - Local vLLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vllm_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
