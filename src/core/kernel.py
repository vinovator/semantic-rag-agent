# Core Kernel Factory
# Manages Semantic Kernel services and database connections
import os
import chromadb
import yaml
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion, GoogleAITextEmbedding

# NEW Imports for v1.0 Vector Store
from semantic_kernel.connectors.chroma import ChromaStore
from src.core.models import KnowledgeRecord 
from src.core.config import load_config

def get_collection():
    """
    Returns a configured Vector Store Collection (The 'Table' in the DB).
    """
    config = load_config()
    
    # 1. Initialize Chroma Vector Store (Persistent)
    persist_path = os.path.join(os.getcwd(), config["chroma"]["persist_path"])
    
    # FIX: Use chromadb client for persistence
    client = chromadb.PersistentClient(path=persist_path)
    vector_store = ChromaStore(client=client)
    
    # 2. Get the Collection with the defined Schema
    collection = vector_store.get_collection(
        collection_name=config["chroma"]["collection_name"],
        record_type=KnowledgeRecord
    )
    return collection

def _create_chat_service(service_config, service_id):
    """
    Factory for creating chat services based on specific config block.
    """
    service_type = service_config["service"]
    
    if service_type == "ollama":
        client = AsyncOpenAI(
            base_url=service_config["endpoint"],
            api_key="ollama"
        )
        return OpenAIChatCompletion(
            ai_model_id=service_config["model_id"],
            async_client=client,
            service_id=service_id
        )
    elif service_type == "gemini":
        api_key = os.environ.get(service_config["api_key_env"])
        return GoogleAIChatCompletion(
            gemini_model_id=service_config["model_id"], 
            api_key=api_key, 
            service_id=service_id
        )
    else:
        raise ValueError(f"Unsupported LLM service type: {service_type}")

def _create_embedding_service(config):
    """
    Helper to create the embedding service. 
    Defaults to the same provider as the 'tools' LLM to keep local with local, or cloud with cloud.
    """
    tools_conf = config.get("tools", {})
    agent_conf = config.get("agent", {})
    
    # Prefer local embeddings if tools are local (save $), otherwise check agent
    target_conf = tools_conf if tools_conf.get("service") == "ollama" else agent_conf
    
    if target_conf.get("service") == "ollama":
         client = AsyncOpenAI(
            base_url=target_conf["endpoint"],
            api_key="ollama"
        )
         return OpenAITextEmbedding(
            ai_model_id="nomic-embed-text", 
            async_client=client
        )
    elif target_conf.get("service") == "gemini":
        api_key = os.environ.get(target_conf["api_key_env"])
        return GoogleAITextEmbedding(embedding_model_id="models/embedding-001", api_key=api_key)
    else:
        # Fallback Default
        raise ValueError("Could not determine embedding service from config.")

def build_kernel():
    config = load_config()
    kernel = Kernel()
    
    # 1. Register Agent Service
    agent_service = _create_chat_service(config["agent"], service_id="agent")
    kernel.add_service(agent_service)
    
    # 2. Register Tools Service
    tool_service = _create_chat_service(config["tools"], service_id="tools")
    kernel.add_service(tool_service)
    
    # 3. Register Embedding Service
    embed_service = _create_embedding_service(config)
    kernel.add_service(embed_service)
    
    return kernel, embed_service