import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """Service class for RAG functionality"""
    
    def __init__(self):
        self.llm = None
        self.model = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.fallback_logs = {
            "vague_answer": 0,
            "no_source_documents": 0,
            "low_semantic_similarity": 0,
            "rag_error": 0,
            "pdf_processing_error": 0
        }
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the RAG service"""
        try:
            # Configure Gemini
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL, 
                google_api_key=config.GOOGLE_API_KEY
            )
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.EMBEDDING_MODEL, 
                google_api_key=config.GOOGLE_API_KEY
            )
            
            # Load and process PDF
            raw_text = self._load_and_process_pdf()
            if not raw_text:
                logger.error("Failed to load and process PDF")
                return False
            
            # Create vector store
            self.vectorstore = self._create_vector_store(raw_text)
            if not self.vectorstore:
                logger.error("Failed to create vector store")
                return False
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={'k': 5, 'fetch_k': 10}
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                memory=self.memory
            )
            
            self._initialized = True
            logger.info("RAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            return False
    
    def _load_and_process_pdf(self) -> Optional[str]:
        """Load and process the constitution PDF"""
        try:
            reader = PdfReader(config.PDF_PATH)
            raw_text = ''.join(
                page.extract_text() for page in reader.pages 
                if page.extract_text()
            )
            
            if not raw_text:
                raise ValueError("Could not extract text from PDF")
                
            logger.info(f"Successfully loaded PDF with {len(raw_text)} characters")
            return raw_text
            
        except FileNotFoundError:
            logger.error(f"PDF file not found at {config.PDF_PATH}")
            self.fallback_logs["pdf_processing_error"] += 1
            return None
        except Exception as e:
            logger.error(f"Error during PDF processing: {e}")
            self.fallback_logs["pdf_processing_error"] += 1
            return None
    
    def _create_vector_store(self, raw_text: str) -> Optional[FAISS]:
        """Create FAISS vector store from text"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            texts = text_splitter.split_text(raw_text)
            
            if not texts:
                raise ValueError("Could not split text into chunks")
                
            vectorstore = FAISS.from_texts(texts, embedding=self.embeddings)
            logger.info(f"Created vector store with {len(texts)} chunks")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None
    
    
    def _format_answer(self, answer: str) -> str:
        """Format the answer for better readability."""
        # Example formatting logic
        formatted_answer = answer.replace('. ', '.\n- ')  # Add bullet points
        formatted_answer = formatted_answer.replace('\n', '\n\n')  # Add extra spacing
        return formatted_answer.strip()

    def ask_question(self, query: str) -> Tuple[str, str, Optional[List[str]], Optional[float]]:
        """
        Ask a question using RAG with fallback to direct LLM
        
        Returns: (answer, source, supporting_articles, semantic_similarity)
        """
        if not self._initialized:
            logger.warning("RAG service not initialized, using direct LLM fallback")
            return self._fallback_to_llm(query), "llm", None, None
        
        try:
            # Use RAG chain
            result = self.qa_chain.invoke({"query": query})
            answer = result["result"].strip()
            source_documents = result["source_documents"]
            
            # Check for vague answers or hallucinations
            hallucination_clues = [
                "i don't know", "i cannot provide", "depends on the jurisdiction",
                "laws vary", "not specified", "not clear", "no specific article",
                "as an ai", "i'm unable", "could not determine", "unclear"
            ]
            
            vague_answer = not answer or any(clue in answer.lower() for clue in hallucination_clues)
            no_sources = not source_documents
            
            # Calculate semantic similarity
            semantic_similarity = -1
            low_similarity = False
            
            try:
                query_embedding = self.embeddings.embed_query(query)
                if answer:
                    answer_embedding = self.embeddings.embed_query(answer)
                    semantic_similarity = cosine_similarity([query_embedding], [answer_embedding])[0][0]
                    low_similarity = semantic_similarity < config.SEMANTIC_SIMILARITY_THRESHOLD
                    logger.info(f"Semantic similarity: {semantic_similarity:.4f}")
                else:
                    low_similarity = True
            except Exception as e:
                logger.warning(f"Error calculating semantic similarity: {e}")
            
            # Check if fallback is needed
            fallback_needed = vague_answer or no_sources or low_similarity
            
            if fallback_needed:
                if vague_answer:
                    self.fallback_logs["vague_answer"] += 1
                elif no_sources:
                    self.fallback_logs["no_source_documents"] += 1
                elif low_similarity:
                    self.fallback_logs["low_semantic_similarity"] += 1
                
                logger.info("Fallback triggered, using direct LLM")
                return self._fallback_to_llm(query), "llm", None, semantic_similarity
            
            # Extract article numbers from source documents
            articles = []
            for doc in source_documents:
                matches = re.findall(r'Article\s*(\d+)', doc.page_content)
                articles.extend(matches)
            
            formatted_answer = self._format_answer(answer)  # Format the answer
            return formatted_answer, "constitution", sorted(set(articles)), semantic_similarity
            
        except Exception as e:
            logger.error(f"Error during RAG process: {e}")
            self.fallback_logs["rag_error"] += 1
            return self._fallback_to_llm(query), "llm", None, None
    
    def _fallback_to_llm(self, query: str) -> str:
        """Fallback to direct LLM when RAG fails"""
        try:
            response = self.model.generate_content(query)
            formatted_answer = self._format_answer(response.text.strip())
            return formatted_answer
        except Exception as e:
            logger.error(f"LLM fallback also failed: {e}")
            return f"❌ Both RAG and LLM failed: {e}"
    
    def get_status(self) -> Dict:
        """Get service status and statistics"""
        return {
            "initialized": self._initialized,
            "fallback_logs": self.fallback_logs,
            "rag_available": self._initialized
        }

# Global RAG service instance
rag_service = RAGService()
