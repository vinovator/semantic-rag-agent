import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.core.agent import RAGAgent

from dotenv import load_dotenv
load_dotenv()

# Global Agent Instance
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    print("--- Booting RAG Agent ---")
    agent_instance = RAGAgent() # Loads LLM, DB, and Models
    yield
    print("--- Shutting Down ---")

app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str
    meta: dict

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not agent_instance:
        raise HTTPException(500, "Agent not initialized")
    
    result = await agent_instance.process_query(request.message)
    
    if "error" in result:
        raise HTTPException(500, result["error"])

    return QueryResponse(
        response=result["answer"],
        meta=result["thought_process"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)