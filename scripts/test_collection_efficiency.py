#!/usr/bin/env python3
"""
Test script for collection querying efficiency improvements.
Demonstrates parallel querying, smart collection filtering, and lazy loading.
"""
import logging
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from database.chroma_client import ChromaDBManager
from conversation.memory import ConversationManager
from rag.query_processing import QueryProcessor
from rag.retrieval import RAGRetriever
from config.settings import OPENAI_API_KEY

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def benchmark_retrieval(retriever, query, collections, num_runs=3):
    """Benchmark the retrieval process."""
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        response, docs, metas = retriever.query_rag(query, collections)
        end_time = time.time()
        
        execution_time = end_time - start_time
        times.append(execution_time)
        
        print(f"Run {i+1}: {execution_time:.2f}s - Retrieved {len(docs)} documents")
    
    avg_time = sum(times) / len(times)
    print(f"Average time over {num_runs} runs: {avg_time:.2f}s")
    return avg_time, times

def test_collection_filtering():
    """Test smart collection filtering."""
    print("\n" + "="*60)
    print("TESTING SMART COLLECTION FILTERING")
    print("="*60)
    
    # Initialize components
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    try:
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
        
        # Test smart filtering with various queries
        test_queries = [
            "Was sind die Bedingungen für die Einbürgerung in der Schweiz?",
            "Welche Aufenthaltsrechte haben Asylsuchende?",
            "Wie funktioniert das schweizerische Steuersystem?",
            "Was sind die Grundrechte in der Bundesverfassung?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            
            # Test with filtering enabled
            retriever_with_filtering = RAGRetriever(llm, embeddings, conversation_manager, query_processor)
            
            # Test collection names caching
            start_time = time.time()
            collection_names = retriever_with_filtering._get_collection_names(collections)
            cache_time = time.time() - start_time
            print(f"   📋 Available collections: {', '.join(collection_names[:3])}...")
            print(f"   ⏱️  Collection names loading time: {cache_time:.3f}s")
            
            # Test filtering (if more than threshold)
            if len(collections) > 3:
                filtered_collections = retriever_with_filtering.collection_filter.filter_collections(
                    query, collections, collection_names
                )
                print(f"   🎯 Filtered to {len(filtered_collections)} collections from {len(collections)}")
                
                # Show which collections were selected
                filtered_names = retriever_with_filtering._get_collection_names(filtered_collections)
                print(f"   ✅ Selected: {', '.join(filtered_names)}")
            else:
                print(f"   ℹ️  Filtering skipped: only {len(collections)} collections (≤ threshold)")
            
    except Exception as e:
        print(f"❌ Error during collection filtering test: {e}")

def test_parallel_vs_sequential():
    """Test parallel vs sequential querying performance."""
    print("\n" + "="*60)
    print("TESTING PARALLEL VS SEQUENTIAL QUERYING")
    print("="*60)
    
    # Initialize components
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    try:
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
        
        # Test query
        test_query = "Was sind die Bedingungen für die Einbürgerung in der Schweiz?"
        print(f"🔍 Test query: {test_query}")
        
        # Temporarily modify config for testing
        import config.settings as settings
        
        # Test sequential querying
        print("\n📈 Testing SEQUENTIAL querying...")
        original_parallel = settings.ENABLE_PARALLEL_QUERYING
        settings.ENABLE_PARALLEL_QUERYING = False
        
        retriever_sequential = RAGRetriever(llm, embeddings, conversation_manager, query_processor)
        sequential_avg, _ = benchmark_retrieval(retriever_sequential, test_query, collections, num_runs=3)
        
        # Test parallel querying
        print("\n🚀 Testing PARALLEL querying...")
        settings.ENABLE_PARALLEL_QUERYING = True
        
        retriever_parallel = RAGRetriever(llm, embeddings, conversation_manager, query_processor)
        parallel_avg, _ = benchmark_retrieval(retriever_parallel, test_query, collections, num_runs=3)
        
        # Restore original setting
        settings.ENABLE_PARALLEL_QUERYING = original_parallel
        
        # Show performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   Sequential: {sequential_avg:.2f}s")
        print(f"   Parallel:   {parallel_avg:.2f}s")
        
        if sequential_avg > parallel_avg:
            speedup = sequential_avg / parallel_avg
            print(f"   🎉 Speedup: {speedup:.2f}x faster with parallel querying!")
        else:
            print(f"   ℹ️  No significant speedup (possibly due to small dataset or overhead)")
            
    except Exception as e:
        print(f"❌ Error during parallel vs sequential test: {e}")

def test_lazy_loading():
    """Test lazy loading of collection metadata."""
    print("\n" + "="*60)
    print("TESTING LAZY LOADING")
    print("="*60)
    
    # Initialize components
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    try:
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
        retriever = RAGRetriever(llm, embeddings, conversation_manager, query_processor)
        
        # Test first load (should populate cache)
        print("\n🔄 First load (populating cache)...")
        start_time = time.time()
        collection_names_1 = retriever._get_collection_names(collections)
        first_load_time = time.time() - start_time
        print(f"   ⏱️  Time: {first_load_time:.3f}s")
        print(f"   📋 Loaded: {', '.join(collection_names_1[:3])}...")
        
        # Test second load (should use cache)
        print("\n⚡ Second load (using cache)...")
        start_time = time.time()
        collection_names_2 = retriever._get_collection_names(collections)
        second_load_time = time.time() - start_time
        print(f"   ⏱️  Time: {second_load_time:.3f}s")
        
        # Verify results are the same
        if collection_names_1 == collection_names_2:
            print("   ✅ Cache consistency verified")
            
            if first_load_time > second_load_time:
                speedup = first_load_time / second_load_time
                print(f"   🎉 Cache speedup: {speedup:.2f}x faster!")
            else:
                print("   ℹ️  Cache overhead minimal (collections might be small)")
        else:
            print("   ❌ Cache inconsistency detected")
            
    except Exception as e:
        print(f"❌ Error during lazy loading test: {e}")

def main():
    """Run all efficiency tests."""
    print("🚀 COLLECTION QUERYING EFFICIENCY TESTS")
    print("="*60)
    
    # Test 1: Smart Collection Filtering
    test_collection_filtering()
    
    # Test 2: Parallel vs Sequential Querying
    test_parallel_vs_sequential()
    
    # Test 3: Lazy Loading
    test_lazy_loading()
    
    print("\n✅ All efficiency tests completed!")
    print("\nTo disable any of these optimizations, edit config/settings.py:")
    print("  - ENABLE_SMART_COLLECTION_FILTERING = False")
    print("  - ENABLE_PARALLEL_QUERYING = False")
    print("  - LAZY_LOADING_ENABLED = False")

if __name__ == "__main__":
    main() 