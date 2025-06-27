"""
Service layer for the Swiss Legal Chatbot RAG system.
"""
import logging

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings import (DEFAULT_MODEL, ENABLE_REFLECTION,
                             MAX_REFLECTION_ITERATIONS, OPENAI_API_KEY,
                             REFLECTION_ADDITIONAL_SOURCES_LIMIT)
from conversation.memory import ConversationManager
from conversation.routing import QueryRouter
from database.chroma_client import ChromaDBManager
from rag.query_processing import QueryProcessor
from rag.reflection import RAGReflector
from rag.retrieval import RAGRetriever


class LegalChatbotService:
    """Main service class that orchestrates the legal chatbot functionality."""
    
    def __init__(self):
        """Initialize the service with all required components."""
        self.initialized = False
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all service components."""
        if not OPENAI_API_KEY:
            return
        
        try:
            # Initialize OpenAI models
            self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=DEFAULT_MODEL)
            
            # Initialize database manager
            self.db_manager = ChromaDBManager()
            self.law_collections = self.db_manager.get_law_collections()
            
            # Initialize conversation and query processing components
            self.conversation_manager = ConversationManager(self.llm)
            self.query_processor = QueryProcessor(self.llm)
            self.query_router = QueryRouter(self.llm)
            
            # Initialize RAG retriever
            self.rag_retriever = RAGRetriever(
                self.llm, 
                self.embeddings, 
                self.conversation_manager, 
                self.query_processor
            )
            
            # Initialize RAG reflector if enabled
            if ENABLE_REFLECTION:
                self.rag_reflector = RAGReflector(
                    self.llm,
                    self.embeddings,
                    self.rag_retriever,
                    max_iterations=MAX_REFLECTION_ITERATIONS
                )
                logging.info("RAG reflection pattern enabled")
            else:
                self.rag_reflector = None
                logging.info("RAG reflection pattern disabled")
            
            # Set up query routing chain
            self._setup_routing_chain()
            
            self.initialized = True
            logging.info("LegalChatbotService initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize LegalChatbotService: {e}")
            self.initialized = False
    
    def _setup_routing_chain(self):
        """Set up the query routing chain."""
        def list_laws_chain(_: dict) -> dict:
            """Chain that gets the loaded law names and formats them."""
            loaded_laws = self.db_manager.get_loaded_law_names(self.law_collections)
            if not loaded_laws:
                response_text = "I currently don't have any specific laws loaded in my database. Please run the ingestion script."
            else:
                law_list = "\n".join([f"- {law_name}" for law_name in loaded_laws])
                response_text = f"Ich kenne folgende Gesetze:\n\n{law_list}"
            
            disclaimer = '\n\nDiese Informationen dienen nur zur allgemeinen Orientierung und stellen keine Rechtsberatung dar. Für konkrete Fälle konsultieren Sie bitte einen qualifizierten Stelle.'
            
            return {
                "response": response_text + disclaimer,
                "sources": {"docs": [], "metas": []},
                "reflection_info": None
            }
        
        def rag_chain_func(input_dict: dict) -> dict:
            """RAG chain function wrapper with optional reflection."""
            query = input_dict["question"]
            conversation_history = st.session_state.get("messages", [])
            
            # Get initial RAG response
            response, retrieved_docs, retrieved_metas = self.rag_retriever.query_rag(
                query, self.law_collections, conversation_history=conversation_history
            )
            
            reflection_info = None
            
            # Apply reflection pattern if enabled
            if self.rag_reflector and ENABLE_REFLECTION:
                try:
                    logging.info("Applying reflection pattern to RAG response")
                    refined_response, reflection_info = self.rag_reflector.reflect_and_refine(
                        original_query=query,
                        initial_response=response,
                        initial_sources=retrieved_docs,
                        initial_metas=retrieved_metas,
                        collections=self.law_collections,
                        conversation_history=conversation_history
                    )
                    response = refined_response
                    logging.info(f"Reflection completed: {reflection_info['final_status']}")
                except Exception as e:
                    logging.error(f"Error during reflection: {e}")
                    # Continue with original response if reflection fails
                    reflection_info = {
                        "error": str(e),
                        "final_status": "reflection_failed"
                    }
            
            return {
                "response": response,
                "sources": {"docs": retrieved_docs, "metas": retrieved_metas},
                "reflection_info": reflection_info
            }
        
        self.routing_chain = self.query_router.create_routing_chain(
            list_laws_chain, rag_chain_func
        )
    
    def is_ready(self):
        """Check if the service is ready to process queries."""
        return self.initialized and self.law_collections
    
    def get_loaded_laws(self):
        """Get list of loaded law names."""
        if not self.initialized:
            return []
        return self.db_manager.get_loaded_law_names(self.law_collections)
    
    def process_query(self, query):
        """
        Process a user query and return the response.
        
        Args:
            query: User's question string
            
        Returns:
            Dictionary with 'response', 'sources', and 'reflection_info' keys
        """
        if not self.is_ready():
            raise Exception("Service not ready. Please check API key and database collections.")
        
        try:
            result = self.routing_chain.invoke({"question": query})
            return result
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            raise
    
    def get_db_manager(self):
        """Get the database manager instance."""
        return self.db_manager
    
    def is_reflection_enabled(self):
        """Check if reflection pattern is enabled."""
        return ENABLE_REFLECTION and self.rag_reflector is not None 