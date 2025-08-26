from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AskRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., description="The question to ask about the constitution")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for multi-turn conversations")

class AskResponse(BaseModel):
    """Response model for answers"""
    answer: str = Field(..., description="The answer to the question")
    source: str = Field(..., description="Source of the answer (constitution or llm)")
    supporting_articles: Optional[List[str]] = Field(None, description="List of supporting article numbers if available")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn conversations")
    semantic_similarity: Optional[float] = Field(None, description="Semantic similarity score if available")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    rag_available: bool = Field(..., description="Whether RAG functionality is available")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ConversationRequest(BaseModel):
    """Request model for multi-turn conversations"""
    messages: List[Dict[str, str]] = Field(..., description="List of message objects with role and content")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")

class ConversationResponse(BaseModel):
    """Response model for multi-turn conversations"""
    response: str = Field(..., description="The response message")
    conversation_id: str = Field(..., description="Conversation ID for continuing the conversation")
