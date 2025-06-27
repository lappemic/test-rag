#!/usr/bin/env python3
"""
Test script for Max-Marginal Relevance (MMR) implementation.
"""
import logging
import time
from typing import List, Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings import OPENAI_API_KEY, MMR_LAMBDA, MMR_FETCH_K, ENABLE_MMR, MMR_USE_FAST_MODE
from conversation.memory import ConversationManager
from database.chroma_client import ChromaDBManager
from rag.query_processing import QueryProcessor
from rag.retrieval import RAGRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)


def test_mmr_vs_traditional():
    """Test MMR vs traditional similarity ranking."""
    print("\n" + "="*60)
    print("TESTING MMR VS TRADITIONAL SIMILARITY RANKING")
    print("="*60)
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    try:
        # Initialize components
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
        
        db_manager = ChromaDBManager()
        collections = db_manager.get_law_collections()
        
        if not collections:
            print("❌ No law collections found. Please run the ingestion script first.")
            return
            
        print(f"📚 Found {len(collections)} law collections")
        
        # Initialize conversation and query processing components
        conversation_manager = ConversationManager(llm)
        query_processor = QueryProcessor(llm)
        
        # Create retriever
        retriever = RAGRetriever(llm, embeddings, conversation_manager, query_processor)
        
        # Test queries
        test_queries = [
            "Was sind die Bedingungen für die Einbürgerung in der Schweiz?",
            "Welche Rechte haben Asylsuchende in der Schweiz?",
            "Wie funktioniert das Schengen-Abkommen für die Schweiz?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 50)
            
            # Test traditional ranking (disable MMR temporarily)
            original_enable_mmr = ENABLE_MMR
            import config.settings as settings
            settings.ENABLE_MMR = False
            
            start_time = time.time()
            response_trad, docs_trad, metas_trad = retriever.query_rag(query, collections, num_results=5)
            trad_time = time.time() - start_time
            
            # Test with MMR enabled
            settings.ENABLE_MMR = True
            
            start_time = time.time()
            response_mmr, docs_mmr, metas_mmr = retriever.query_rag(query, collections, num_results=5)
            mmr_time = time.time() - start_time
            
            # Restore original setting
            settings.ENABLE_MMR = original_enable_mmr
            
            # Compare results
            print(f"📊 RESULTS COMPARISON:")
            print(f"   ⏱️  Traditional ranking time: {trad_time:.3f}s")
            print(f"   ⏱️  MMR ranking time: {mmr_time:.3f}s")
            print(f"   📈 Speed overhead: {((mmr_time - trad_time) / trad_time * 100):.1f}%")
            
            print(f"\n   📋 Traditional ranking sources:")
            for i, meta in enumerate(metas_trad[:3]):
                doc_title = meta.get('document_title', 'Unknown')
                article = meta.get('article_id', 'N/A')
                print(f"      {i+1}. {doc_title} - {article}")
            
            print(f"\n   🎯 MMR ranking sources:")
            for i, meta in enumerate(metas_mmr[:3]):
                doc_title = meta.get('document_title', 'Unknown')
                article = meta.get('article_id', 'N/A')
                print(f"      {i+1}. {doc_title} - {article}")
            
            # Calculate diversity metrics
            trad_sources = set(meta.get('document_title', 'Unknown') for meta in metas_trad)
            mmr_sources = set(meta.get('document_title', 'Unknown') for meta in metas_mmr)
            
            print(f"\n   📊 Diversity metrics:")
            print(f"      Traditional: {len(trad_sources)} unique sources out of {len(metas_trad)}")
            print(f"      MMR: {len(mmr_sources)} unique sources out of {len(metas_mmr)}")
            
            if len(mmr_sources) > len(trad_sources):
                print(f"      ✅ MMR achieved better diversity (+{len(mmr_sources) - len(trad_sources)} unique sources)")
            elif len(mmr_sources) == len(trad_sources):
                print(f"      ➡️  Same diversity level")
            else:
                print(f"      ❌ MMR achieved lower diversity (-{len(trad_sources) - len(mmr_sources)} unique sources)")
    
    except Exception as e:
        print(f"❌ Error during MMR testing: {e}")


def test_mmr_parameters():
    """Test different MMR parameter values."""
    print("\n" + "="*60)
    print("TESTING MMR PARAMETER SENSITIVITY")
    print("="*60)
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    try:
        # Initialize components
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
        
        db_manager = ChromaDBManager()
        collections = db_manager.get_law_collections()
        
        if not collections:
            print("❌ No law collections found. Please run the ingestion script first.")
            return
        
        # Initialize conversation and query processing components
        conversation_manager = ConversationManager(llm)
        query_processor = QueryProcessor(llm)
        
        # Create retriever
        retriever = RAGRetriever(llm, embeddings, conversation_manager, query_processor)
        
        test_query = "Was sind die Bedingungen für die Einbürgerung in der Schweiz?"
        lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]  # From max diversity to max relevance
        
        print(f"🔍 Test query: {test_query}")
        print(f"📊 Testing λ values: {lambda_values}")
        print("   (λ=0.0: max diversity, λ=1.0: max relevance)")
        
        import config.settings as settings
        original_lambda = settings.MMR_LAMBDA
        original_enable_mmr = settings.ENABLE_MMR
        settings.ENABLE_MMR = True
        
        for lambda_val in lambda_values:
            settings.MMR_LAMBDA = lambda_val
            
            response, docs, metas = retriever.query_rag(test_query, collections, num_results=5)
            
            # Calculate diversity
            sources = set(meta.get('document_title', 'Unknown') for meta in metas)
            articles = set(meta.get('article_id', 'N/A') for meta in metas)
            
            print(f"\n   λ={lambda_val}: {len(sources)} unique sources, {len(articles)} unique articles")
            for i, meta in enumerate(metas[:3]):
                doc_title = meta.get('document_title', 'Unknown')[:30] + "..."
                article = meta.get('article_id', 'N/A')
                print(f"      {i+1}. {doc_title} - {article}")
        
        # Restore original settings
        settings.MMR_LAMBDA = original_lambda
        settings.ENABLE_MMR = original_enable_mmr
        
    except Exception as e:
        print(f"❌ Error during parameter testing: {e}")


def print_mmr_config():
    """Print current MMR configuration."""
    print("\n" + "="*60)
    print("CURRENT MMR CONFIGURATION")
    print("="*60)
    
    print(f"📋 MMR Settings:")
    print(f"   ENABLE_MMR: {ENABLE_MMR}")
    print(f"   MMR_LAMBDA: {MMR_LAMBDA} (relevance vs diversity trade-off)")
    print(f"   MMR_FETCH_K: {MMR_FETCH_K} (documents to fetch before re-ranking)")
    print(f"   MMR_USE_FAST_MODE: {MMR_USE_FAST_MODE} (heuristic vs embedding-based)")
    
    print(f"\n📝 Interpretation:")
    print(f"   - λ closer to 0.0 = more diverse results")
    print(f"   - λ closer to 1.0 = more relevant results")
    print(f"   - Fast mode uses heuristics (faster, less accurate)")
    print(f"   - Full mode uses embeddings (slower, more accurate)")


if __name__ == "__main__":
    print("🧪 MMR Testing Suite")
    
    print_mmr_config()
    test_mmr_vs_traditional()
    test_mmr_parameters()
    
    print(f"\n✅ MMR testing completed!")
    print(f"💡 You can adjust MMR parameters in config/settings.py:")
    print(f"   - MMR_LAMBDA: balance relevance vs diversity")
    print(f"   - MMR_FETCH_K: number of candidates to consider")
    print(f"   - MMR_USE_FAST_MODE: speed vs accuracy trade-off") 