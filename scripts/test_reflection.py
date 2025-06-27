#!/usr/bin/env python3
"""
Test script for the reflection pattern implementation.
"""

import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings import DEFAULT_MODEL, OPENAI_API_KEY
from rag.reflection import EvaluationStatus, RAGReflector


def test_reflection_evaluation():
    """Test the reflection evaluation functionality."""
    print("Testing Reflection Pattern Evaluation...")
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    # Initialize components
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=DEFAULT_MODEL)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Create a mock RAG retriever for testing
    class MockRAGRetriever:
        def query_rag(self, query, collections, num_results=3, conversation_history=None):
            # Return some mock results
            return f"Mock response for: {query}", [f"Mock document for {query}"], [{"document_title": "Test Law", "article_id": "Art. 1"}]
    
    mock_retriever = MockRAGRetriever()
    
    # Initialize reflector
    reflector = RAGReflector(llm, embeddings, mock_retriever, max_iterations=2)
    
    # Test case 1: Response that should pass (no additional sources needed)
    print("\n🧪 Test 1: Response that should pass")
    complete_response = """
    According to Article 3 of the Asylum Law (AsylG), SR 142.20, asylum seekers must apply for asylum within a specific timeframe. 
    The law clearly states the requirements and procedures for asylum applications.
    
    Diese Informationen dienen nur zur allgemeinen Orientierung und stellen keine Rechtsberatung dar.
    """
    
    evaluation1 = reflector.evaluate_response(
        "What are the requirements for asylum applications?",
        complete_response,
        ["Asylum Law (AsylG), SR 142.20, Article 3"]
    )
    
    print(f"Status: {evaluation1.evaluation.value}")
    print(f"Feedback: {evaluation1.feedback}")
    print(f"Mentioned sources: {evaluation1.mentioned_sources}")
    
    # Test case 2: Response that mentions additional sources
    print("\n🧪 Test 2: Response that mentions additional sources")
    incomplete_response = """
    According to Article 3 of the Asylum Law (AsylG), asylum seekers must apply for asylum within a specific timeframe. 
    However, this should be read in conjunction with the Foreign Nationals and Integration Act (AIG) and the 
    specific provisions in Article 12 of the Federal Constitution regarding fundamental rights.
    
    Diese Informationen dienen nur zur allgemeinen Orientierung und stellen keine Rechtsberatung dar.
    """
    
    evaluation2 = reflector.evaluate_response(
        "What are the requirements for asylum applications?",
        incomplete_response,
        ["Asylum Law (AsylG), SR 142.20, Article 3"]
    )
    
    print(f"Status: {evaluation2.evaluation.value}")
    print(f"Feedback: {evaluation2.feedback}")
    print(f"Mentioned sources: {evaluation2.mentioned_sources}")
    
    # Test case 3: Test refinement
    print("\n🧪 Test 3: Testing response refinement")
    
    additional_docs = ["Article 12 of the Federal Constitution guarantees fundamental rights..."]
    additional_metas = [{"document_title": "Federal Constitution", "article_id": "Art. 12"}]
    
    refined_response = reflector.refine_response(
        "What are the requirements for asylum applications?",
        incomplete_response,
        additional_docs,
        additional_metas,
        "Include information about constitutional rights"
    )
    
    print(f"Original response length: {len(incomplete_response)}")
    print(f"Refined response length: {len(refined_response)}")
    print(f"Refinement successful: {len(refined_response) > len(incomplete_response)}")
    
    print("\n✅ Reflection pattern tests completed!")

if __name__ == "__main__":
    test_reflection_evaluation() 