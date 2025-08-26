from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from contextlib import asynccontextmanager
import uvicorn
from typing import Dict, Any
import os

from config import config
from rag_service import rag_service
from models import (
    AskRequest, AskResponse, HealthResponse, 
    ErrorResponse, ConversationRequest, ConversationResponse
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI application"""
    # Startup: Initialize RAG service
    print("Initializing RAG service...")
    rag_initialized = rag_service.initialize()
    if rag_initialized:
        print("RAG service initialized successfully")
    else:
        print("RAG service initialization failed - will use LLM fallback only")
    
    yield
    
    # Shutdown: Clean up resources
    print("Shutting down application...")

# Create templates directory and setup
templates = Jinja2Templates(directory="templates")

# Create FastAPI application
app = FastAPI(
    title="Constitution Chatbot API",
    description="API for querying the Indian Constitution using RAG with Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (if any)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the frontend interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Constitution Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    status = rag_service.get_status()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        rag_available=status["rag_available"]
    )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question about the constitution"""
    try:
        answer, source, articles, similarity = rag_service.ask_question(request.question)
        
        return AskResponse(
            answer=answer,
            source=source,
            supporting_articles=articles,
            conversation_id=request.conversation_id,
            semantic_similarity=similarity
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )

@app.post("/conversation", response_model=ConversationResponse)
async def conversation(request: ConversationRequest):
    """Multi-turn conversation endpoint"""
    try:
        # For now, we'll use the same logic as /ask but can be extended
        # for proper conversation management
        question = request.messages[-1]["content"] if request.messages else ""
        
        answer, source, articles, similarity = rag_service.ask_question(question)
        
        return ConversationResponse(
            response=answer,
            conversation_id=request.conversation_id or "default-conversation"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing conversation: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get service statistics and fallback logs"""
    status = rag_service.get_status()
    return {
        "rag_initialized": status["initialized"],
        "fallback_stats": status["fallback_logs"],
        "rag_available": status["rag_available"]
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return ErrorResponse(
        error=str(exc.detail),
        details={"status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return ErrorResponse(
        error="Internal server error",
        details={"exception": str(exc)}
    )

if __name__ == "__main__":
    """Run the application"""
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )
