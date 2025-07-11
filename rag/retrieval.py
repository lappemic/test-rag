"""
Document retrieval and RAG processing for the Swiss Legal Chatbot.
"""
import concurrent.futures
import logging
import operator
from typing import Dict, List, Optional, Tuple, Callable, Generator, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config.settings import (COLLECTION_FILTERING_THRESHOLD,
                             ENABLE_PARALLEL_QUERYING,
                             ENABLE_SMART_COLLECTION_FILTERING,
                             LAZY_LOADING_ENABLED, MAX_PARALLEL_WORKERS,
                             MAX_RESULTS, ENABLE_STREAMING, STREAMING_CHUNK_SIZE,
                             ENABLE_MMR, MMR_LAMBDA, MMR_FETCH_K, MMR_USE_FAST_MODE)


class CollectionFilter:
    """Handles smart filtering of collections based on query intent."""
    
    def __init__(self, llm):
        """Initialize with language model for collection filtering."""
        self.llm = llm
        self._setup_filter_prompt()
    
    def _setup_filter_prompt(self):
        """Set up the collection filtering prompt."""
        self.filter_prompt = ChatPromptTemplate.from_template(
            """Du bist ein Experte für schweizerisches Recht. Gegeben ist eine Benutzeranfrage und eine Liste verfügbarer Rechtssammlungen.

            Deine Aufgabe ist es, die 3 relevantesten Rechtssammlungen für diese Anfrage zu identifizieren.

            VERFÜGBARE RECHTSSAMMLUNGEN:
            {available_collections}

            BENUTZERANFRAGE: {query}

            Analysiere die Anfrage und bestimme, welche Rechtssammlungen am wahrscheinlichsten relevante Informationen enthalten.

            Antworte nur mit einer kommagetrennten Liste der relevantesten Sammlungsnamen (maximal 3), z.B.:
            Asylgesetz (AsylG), Ausländer- und Integrationsgesetz (AIG), Schweizerische Bundesverfassung (BV)

            Wenn die Anfrage sehr allgemein ist oder mehrere Rechtsbereiche betrifft, wähle die wichtigsten Grundgesetze."""
                    )
    
    def filter_collections(self, query: str, collections: List, collection_names: List[str]) -> List:
        """
        Filter collections based on query intent.
        
        Args:
            query: User's question
            collections: List of ChromaDB collection objects
            collection_names: List of law names corresponding to collections
            
        Returns:
            List of filtered collection objects
        """
        if len(collections) <= 3:
            # If we have 3 or fewer collections, use all
            return collections
        
        try:
            # Create mapping between collection names and objects
            name_to_collection = {}
            for collection, name in zip(collections, collection_names):
                name_to_collection[name] = collection
            
            available_collections_str = "\n".join([f"- {name}" for name in collection_names])
            
            filter_chain = self.filter_prompt | self.llm | StrOutputParser()
            result = filter_chain.invoke({
                "query": query,
                "available_collections": available_collections_str
            })
            
            # Parse the result to get collection names
            selected_names = [name.strip() for name in result.split(",")]
            selected_names = [name for name in selected_names if name in collection_names]
            
            # Get corresponding collection objects
            filtered_collections = []
            for name in selected_names[:3]:  # Limit to 3
                if name in name_to_collection:
                    filtered_collections.append(name_to_collection[name])
            
            if filtered_collections:
                logging.info(f"Filtered to {len(filtered_collections)} collections: {selected_names[:len(filtered_collections)]}")
                return filtered_collections
            else:
                logging.warning("Collection filtering failed, using all collections")
                return collections
                
        except Exception as e:
            logging.warning(f"Collection filtering failed: {e}. Using all collections.")
            return collections


class ParallelCollectionQuerier:
    """Handles parallel querying of multiple collections."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize with maximum number of worker threads."""
        self.max_workers = max_workers
    
    def query_collection(self, collection, query_embedding: List[float], num_results: int) -> List[Dict]:
        """
        Query a single collection and return formatted results.
        
        Args:
            collection: ChromaDB collection object
            query_embedding: Query embedding vector
            num_results: Number of results to retrieve
            
        Returns:
            List of result dictionaries
        """
        try:
            results = collection.query(
                query_embeddings=[query_embedding], 
                n_results=num_results,
                include=["metadatas", "documents", "distances"]
            )
            
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i],
                    "collection_name": collection.name
                })
            
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error querying collection {collection.name}: {e}")
            return []
    
    def query_collections_parallel(self, collections: List, query_embedding: List[float], num_results: int) -> List[Dict]:
        """
        Query multiple collections in parallel.
        
        Args:
            collections: List of ChromaDB collection objects
            query_embedding: Query embedding vector
            num_results: Number of results to retrieve per collection
            
        Returns:
            List of all results from all collections
        """
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all collection queries concurrently
            future_to_collection = {
                executor.submit(self.query_collection, collection, query_embedding, num_results): collection
                for collection in collections
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_collection):
                collection = future_to_collection[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logging.debug(f"Retrieved {len(results)} results from collection {collection.name}")
                except Exception as e:
                    logging.error(f"Error processing results from collection {collection.name}: {e}")
        
        return all_results


class RAGRetriever:
    """Handles document retrieval and RAG processing with optimized collection querying."""
    
    def __init__(self, llm, embeddings, conversation_manager, query_processor):
        """
        Initialize with required components.
        
        Args:
            llm: Language model for response generation
            embeddings: Embedding model for query processing
            conversation_manager: ConversationManager instance
            query_processor: QueryProcessor instance
        """
        self.llm = llm
        self.embeddings = embeddings
        self.conversation_manager = conversation_manager
        self.query_processor = query_processor
        
        # Initialize optimization components based on configuration
        self.collection_filter = CollectionFilter(llm) if ENABLE_SMART_COLLECTION_FILTERING else None
        self.parallel_querier = ParallelCollectionQuerier(max_workers=MAX_PARALLEL_WORKERS) if ENABLE_PARALLEL_QUERYING else None
        
        self._setup_prompt()
        
        # Cache for collection names to avoid repeated metadata queries (lazy loading)
        self._collection_names_cache = {} if LAZY_LOADING_ENABLED else None
    
    def _get_collection_names(self, collections: List) -> List[str]:
        """
        Get law names for collections with optional caching (lazy loading).
        
        Args:
            collections: List of ChromaDB collection objects
            
        Returns:
            List of law names corresponding to collections
        """
        collection_names = []
        
        for collection in collections:
            # Check cache if lazy loading is enabled
            if LAZY_LOADING_ENABLED and self._collection_names_cache and collection.name in self._collection_names_cache:
                collection_names.append(self._collection_names_cache[collection.name])
            else:
                try:
                    # Get one item to find the document title
                    meta = collection.get(limit=1, include=["metadatas"])
                    if meta["metadatas"]:
                        doc_title = meta["metadatas"][0].get("document_title", "Unknown")
                        # Cache the result if lazy loading is enabled
                        if LAZY_LOADING_ENABLED and self._collection_names_cache is not None:
                            self._collection_names_cache[collection.name] = doc_title
                        collection_names.append(doc_title)
                    else:
                        collection_names.append("Unknown")
                except Exception as e:
                    logging.error(f"Error getting metadata from collection {collection.name}: {e}")
                    collection_names.append("Unknown")
        
        return collection_names
    
    def _query_collections_sequential(self, collections: List, query_embedding: List[float], num_results: int) -> List[Dict]:
        """
        Query collections sequentially (fallback when parallel querying is disabled).
        
        Args:
            collections: List of ChromaDB collection objects
            query_embedding: Query embedding vector
            num_results: Number of results to retrieve per collection
            
        Returns:
            List of all results from all collections
        """
        all_results = []
        
        for collection in collections:
            try:
                results = collection.query(
                    query_embeddings=[query_embedding], 
                    n_results=num_results,
                    include=["metadatas", "documents", "distances"]
                )
                
                # Format results to match parallel querier output
                for i in range(len(results["ids"][0])):
                    all_results.append({
                        "id": results["ids"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "document": results["documents"][0][i],
                        "collection_name": collection.name
                    })
                
                logging.debug(f"Retrieved {len(results['ids'][0])} results from collection {collection.name}")
                
            except Exception as e:
                logging.error(f"Error querying collection {collection.name}: {e}")
        
        return all_results

    def _apply_mmr_reranking(self, query_embedding: List[float], all_results: List[Dict], num_results: int) -> List[Dict]:
        """
        Apply Max-Marginal Relevance (MMR) re-ranking to improve diversity in retrieved chunks.
        
        MMR balances relevance to the query with diversity among selected documents.
        Formula: MMR(d) = λ * Sim(d, query) - (1-λ) * max(Sim(d, d_i)) for d_i in selected
        
        Args:
            query_embedding: Query embedding vector
            all_results: List of all retrieved results with embeddings/distances
            num_results: Final number of results to return
            
        Returns:
            List of re-ranked results with improved diversity
        """
        if not ENABLE_MMR or len(all_results) <= num_results:
            # If MMR is disabled or we have fewer results than needed, return as-is
            return all_results[:num_results]
        
        logging.info(f"Applying MMR re-ranking to {len(all_results)} results (λ={MMR_LAMBDA})")
        
        try:
            # Convert distances to similarities (ChromaDB returns cosine distances, sim = 1 - distance)
            similarities_to_query = [1.0 - result["distance"] for result in all_results]
            
            # Get document texts for embedding
            document_texts = [result["document"] for result in all_results]
            
            # Compute embeddings for all documents (for accurate similarity calculation)
            # This is more expensive but gives better diversity results
            logging.debug("Computing document embeddings for MMR...")
            document_embeddings = []
            
            # Batch embedding computation for efficiency
            batch_size = 10  # Process documents in batches to avoid memory issues
            for i in range(0, len(document_texts), batch_size):
                batch_texts = document_texts[i:i + batch_size]
                batch_embeddings = [self.embeddings.embed_query(text) for text in batch_texts]
                document_embeddings.extend(batch_embeddings)
            
            # Convert to numpy arrays for efficient computation
            query_emb = np.array(query_embedding).reshape(1, -1)
            doc_embs = np.array(document_embeddings)
            
            # MMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(all_results)))
            
            # Start with the most relevant document (first in sorted order)
            best_idx = remaining_indices[0]  
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Select remaining documents using MMR
            for _ in range(min(num_results - 1, len(remaining_indices))):
                best_score = -float('inf')
                best_idx = None
                
                for idx in remaining_indices:
                    # Relevance component (similarity to query)
                    relevance = similarities_to_query[idx]
                    
                    # Diversity component (max similarity to already selected documents)
                    max_sim_to_selected = 0.0
                    if selected_indices:
                        # Calculate cosine similarity between current doc and all selected docs
                        current_doc_emb = doc_embs[idx:idx+1]
                        selected_doc_embs = doc_embs[selected_indices]
                        
                        # Compute cosine similarity
                        similarities = cosine_similarity(current_doc_emb, selected_doc_embs)[0]
                        max_sim_to_selected = np.max(similarities)
                    
                    # MMR score
                    mmr_score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim_to_selected
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
            
            # Return results in MMR order
            mmr_results = [all_results[idx] for idx in selected_indices]
            
            logging.info(f"MMR re-ranking completed: selected {len(mmr_results)} diverse results")
            return mmr_results
            
        except Exception as e:
            logging.warning(f"MMR re-ranking failed: {e}. Falling back to similarity ranking.")
            return all_results[:num_results]

    def _apply_fast_mmr_reranking(self, query_embedding: List[float], all_results: List[Dict], num_results: int) -> List[Dict]:
        """
        Apply fast MMR re-ranking using heuristic-based similarity approximation.
        
        This version uses query similarity scores to approximate document-to-document similarity
        for faster computation when you need reasonable diversity without the embedding overhead.
        
        Args:
            query_embedding: Query embedding vector  
            all_results: List of all retrieved results
            num_results: Final number of results to return
            
        Returns:
            List of re-ranked results with improved diversity
        """
        if not ENABLE_MMR or len(all_results) <= num_results:
            return all_results[:num_results]
        
        logging.info(f"Applying fast MMR re-ranking to {len(all_results)} results (heuristic-based)")
        
        try:
            # Convert distances to similarities
            similarities_to_query = [1.0 - result["distance"] for result in all_results]
            
            selected_indices = []
            remaining_indices = list(range(len(all_results)))
            
            # Start with the most relevant document
            best_idx = remaining_indices[0]  
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Select remaining documents using heuristic MMR
            for _ in range(min(num_results - 1, len(remaining_indices))):
                best_score = -float('inf')
                best_idx = None
                
                for idx in remaining_indices:
                    # Relevance component
                    relevance = similarities_to_query[idx]
                    
                    # Diversity component (heuristic approximation)
                    max_sim_to_selected = 0.0
                    for selected_idx in selected_indices:
                        # Approximate similarity using query similarity differences
                        # Documents with similar query relevance are likely similar to each other
                        sim_diff = abs(similarities_to_query[idx] - similarities_to_query[selected_idx])
                        doc_similarity = max(0.0, 1.0 - sim_diff)  # Convert difference to similarity
                        max_sim_to_selected = max(max_sim_to_selected, doc_similarity)
                    
                    # MMR score
                    mmr_score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim_to_selected
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
            
            mmr_results = [all_results[idx] for idx in selected_indices]
            logging.info(f"Fast MMR re-ranking completed: selected {len(mmr_results)} diverse results")
            return mmr_results
            
        except Exception as e:
            logging.warning(f"Fast MMR re-ranking failed: {e}. Falling back to similarity ranking.")
            return all_results[:num_results]

    def _setup_prompt(self):
        """Set up the RAG prompt template with system message and dynamic prompt."""
        # Fixed system instructions that don't change per query
        self.system_message = """
        You are a Swiss legal expert chatbot with comprehensive knowledge of Swiss federal laws, regulations, and legal principles, specializing in the Systematic Recompilation (SR) of Swiss legislation.

        CORE INSTRUCTIONS:

        **Accuracy and Citations**:
        - Base responses exclusively on provided context from Swiss legal documents
        - Cite sources using format: `[Document Title, SR Number, Article ID/Paragraph ID, Date of Applicability]`
        - Example: `[Asylgesetz (AsylG), SR 142.20, Art. 3/Para. 1, 2025-01-01]`
        - Prioritize most specific and recent provisions based on `date_applicability`

        **Response Structure**:
        - Organize responses logically using bullet points, numbered lists, or headings
        - Break down complex legal concepts for non-experts while maintaining precision
        - Define technical terms (e.g., "SR Number" as Systematic Recompilation Number)

        **Metadata Usage**:
        - Reference source chunks using metadata fields (`document_title`, `sr_number`, `article_id`, `paragraph_id`)
        - Include `date_applicability` for temporal validity
        - Incorporate `references` or `amendment_history` when available

        **Tone and Language**:
        - Maintain objective, impartial tone
        - Respond in German (language of context) unless user specifies otherwise
        - Never speculate beyond provided context

        **Legal Disclaimer**:
        - Include in every response: "Diese Informationen dienen nur zur allgemeinen Orientierung und stellen keine Rechtsberatung dar. Für konkrete Fälle konsultieren Sie bitte einen qualifizierten Stelle.\"
        """

        # Shortened dynamic prompt template for context-dependent instructions
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("user", """
            {conversation_context_section}

            CONTEXT-SPECIFIC GUIDELINES:

            1. **Conversation Awareness** (if conversation history provided):
            - Consider conversation history when formulating your response
            - Reference previous topics when relevant and maintain continuity

            2. **Comprehensive Analysis**:
            - Apply cited provisions to the specific question/scenario
            - Address potential ambiguities or exceptions within the context
            - Provide reasoned analysis grounded in the legal text

            3. **Handling Insufficient Context**:
            - If context is insufficient: "The provided context does not contain sufficient information to fully address the question."
            - Offer general legal principles if applicable, noting limitations
            - Suggest potential sources when relevant

            4. **Known Laws Queries**:
            - If asked about known laws, list unique `document_title` values with `sr_number`
            - Format: "Ich kenne folgende Gesetze: [Law Name, SR Number], ..."

            5. **Error Handling**:
            - If contradictory chunks exist, prioritize most recent and specific provision
            - If no relevant chunks found: "No relevant legal provisions were found in the provided context for this question."

            LEGAL CONTEXT:
            {context}

            CURRENT QUESTION: {question}
            """)
                    ])
    
    def query_rag(self, query, collections, num_results=None, conversation_history=None):
        """
        Query the RAG system with conversation memory and optimized collection querying.
        
        Args:
            query: User's question
            collections: ChromaDB collections to search
            num_results: Number of results to retrieve
            conversation_history: List of previous messages
        
        Returns:
            Tuple of (response, retrieved_docs, retrieved_metas)
        """
        if num_results is None:
            num_results = MAX_RESULTS
            
        logging.info(f"Querying RAG system with query: {query}")
        
        # Build conversation context
        conversation_context = self.conversation_manager.build_conversation_context(
            conversation_history or []
        )
        
        # Enhance query with conversation context for better retrieval
        enhanced_query = self.query_processor.enhance_query_with_context(
            query, conversation_context
        )
        
        # Use enhanced query for embedding and retrieval
        query_embedding = self.embeddings.embed_query(enhanced_query)
        
        # Optimize collection selection based on configuration
        if ENABLE_SMART_COLLECTION_FILTERING and self.collection_filter and len(collections) > COLLECTION_FILTERING_THRESHOLD:
            # Get collection names (with optional caching)
            collection_names = self._get_collection_names(collections)
            
            # Filter collections based on query intent
            filtered_collections = self.collection_filter.filter_collections(
                enhanced_query, collections, collection_names
            )
            logging.info(f"Smart filtering enabled: querying {len(filtered_collections)} collections (filtered from {len(collections)})")
        else:
            filtered_collections = collections
            if ENABLE_SMART_COLLECTION_FILTERING:
                logging.info(f"Smart filtering skipped: {len(collections)} collections <= threshold ({COLLECTION_FILTERING_THRESHOLD})")
        
        # Determine number of results to fetch for MMR
        fetch_k = MMR_FETCH_K if ENABLE_MMR else num_results
        
        # Query collections (parallel or sequential based on configuration)
        if ENABLE_PARALLEL_QUERYING and self.parallel_querier:
            logging.info(f"Using parallel querying with {MAX_PARALLEL_WORKERS} workers")
            all_results = self.parallel_querier.query_collections_parallel(
                filtered_collections, query_embedding, fetch_k
            )
        else:
            logging.info("Using sequential querying")
            all_results = self._query_collections_sequential(filtered_collections, query_embedding, fetch_k)
        
        # Sort all results by distance (ascending)
        sorted_results = sorted(all_results, key=operator.itemgetter('distance'))
        
        # Apply MMR re-ranking for diverse results
        if ENABLE_MMR and len(sorted_results) > num_results:
            logging.info("Applying MMR re-ranking for improved diversity")
            if MMR_USE_FAST_MODE:
                top_results = self._apply_fast_mmr_reranking(query_embedding, sorted_results, num_results)
            else:
                top_results = self._apply_mmr_reranking(query_embedding, sorted_results, num_results)
        else:
            # Take the top N results overall (traditional approach)
            top_results = sorted_results[:num_results]

        retrieved_docs = [res["document"] for res in top_results]
        retrieved_metas = [res["metadata"] for res in top_results]
        
        context = "\n".join([
            f"[Source: {meta.get('document_title', 'Unknown')} - {meta.get('article_id', 'Unknown') or 'meta'}]\n{doc}"
            for doc, meta in zip(retrieved_docs, retrieved_metas)
        ])
        
        logging.info(f"Retrieved {len(retrieved_docs)} documents for query across {len(filtered_collections)} collections.")
        
        # Format conversation context for the prompt
        conversation_context_section = ""
        if conversation_context:
            conversation_context_section = f"""
            CONVERSATION HISTORY:
            {conversation_context}

            Please consider this conversation history when answering the current question. Reference topics when relevant and maintain continuity in your responses.
            """
        
        rag_chain = (
            {
                "context": RunnablePassthrough(), 
                "question": RunnablePassthrough(),
                "conversation_context_section": lambda x: conversation_context_section
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        response = rag_chain.invoke({"context": context, "question": query})
        logging.info("Generated response from RAG chain with conversation context")
        return response, retrieved_docs, retrieved_metas 

    def query_rag_streaming(self, query, collections, num_results=None, conversation_history=None, stage_callback: Callable[[str, str], None] = None) -> Tuple[Generator[str, Any, None], List[str], List[Dict]]:
        """
        Query the RAG system with conversation memory and optimized collection querying, supporting streaming.
        
        Args:
            query: User's question
            collections: ChromaDB collections to search
            num_results: Number of results to retrieve
            conversation_history: List of previous messages
            stage_callback: Callback function to be called at each stage of the process
        
        Returns:
            Tuple of (response_generator, retrieved_docs, retrieved_metas)
        """
        if num_results is None:
            num_results = MAX_RESULTS
            
        logging.info(f"Querying RAG system with streaming enabled: {query}")
        
        # Stage 1: Building conversation context
        if stage_callback:
            stage_callback("context", "Aufbau des Gesprächskontexts...")
        
        conversation_context = self.conversation_manager.build_conversation_context(
            conversation_history or []
        )
        
        # Stage 2: Query enhancement
        if stage_callback:
            stage_callback("enhancement", "Verbesserung der Anfrage...")
        
        enhanced_query = self.query_processor.enhance_query_with_context(
            query, conversation_context
        )
        
        # Stage 3: Creating embeddings
        if stage_callback:
            stage_callback("embedding", "Erstelle Einbettungen...")
        
        query_embedding = self.embeddings.embed_query(enhanced_query)
        
        # Stage 4: Collection filtering
        if stage_callback:
            stage_callback("filtering", "Filtere relevante Rechtssammlungen...")
        
        if ENABLE_SMART_COLLECTION_FILTERING and self.collection_filter and len(collections) > COLLECTION_FILTERING_THRESHOLD:
            collection_names = self._get_collection_names(collections)
            filtered_collections = self.collection_filter.filter_collections(
                enhanced_query, collections, collection_names
            )
            logging.info(f"Smart filtering enabled: querying {len(filtered_collections)} collections (filtered from {len(collections)})")
        else:
            filtered_collections = collections
            if ENABLE_SMART_COLLECTION_FILTERING:
                logging.info(f"Smart filtering skipped: {len(collections)} collections <= threshold ({COLLECTION_FILTERING_THRESHOLD})")
        
        # Stage 5: Document retrieval
        if stage_callback:
            stage_callback("retrieval", f"Durchsuche {len(filtered_collections)} Rechtssammlungen...")
        
        # Determine number of results to fetch for MMR
        fetch_k = MMR_FETCH_K if ENABLE_MMR else num_results
        
        if ENABLE_PARALLEL_QUERYING and self.parallel_querier:
            logging.info(f"Using parallel querying with {MAX_PARALLEL_WORKERS} workers")
            all_results = self.parallel_querier.query_collections_parallel(
                filtered_collections, query_embedding, fetch_k
            )
        else:
            logging.info("Using sequential querying")
            all_results = self._query_collections_sequential(filtered_collections, query_embedding, fetch_k)
        
        # Stage 6: Result processing and MMR re-ranking
        if stage_callback:
            if ENABLE_MMR:
                stage_callback("processing", "Verarbeite und diversifiziere gefundene Dokumente...")
            else:
                stage_callback("processing", "Verarbeite gefundene Dokumente...")
        
        sorted_results = sorted(all_results, key=operator.itemgetter('distance'))
        
        # Apply MMR re-ranking for diverse results
        if ENABLE_MMR and len(sorted_results) > num_results:
            logging.info("Applying MMR re-ranking for improved diversity")
            if MMR_USE_FAST_MODE:
                top_results = self._apply_fast_mmr_reranking(query_embedding, sorted_results, num_results)
            else:
                top_results = self._apply_mmr_reranking(query_embedding, sorted_results, num_results)
        else:
            # Take the top N results overall (traditional approach)
            top_results = sorted_results[:num_results]

        retrieved_docs = [res["document"] for res in top_results]
        retrieved_metas = [res["metadata"] for res in top_results]
        
        context = "\n".join([
            f"[Source: {meta.get('document_title', 'Unknown')} - {meta.get('article_id', 'Unknown') or 'meta'}]\n{doc}"
            for doc, meta in zip(retrieved_docs, retrieved_metas)
        ])
        
        logging.info(f"Retrieved {len(retrieved_docs)} documents for query across {len(filtered_collections)} collections.")
        
        # Stage 7: Response generation
        if stage_callback:
            stage_callback("generation", "Generiere Antwort...")
        
        # Format conversation context for the prompt
        conversation_context_section = ""
        if conversation_context:
            conversation_context_section = f"""
            CONVERSATION HISTORY:
            {conversation_context}

            Please consider this conversation history when answering the current question. Reference topics when relevant and maintain continuity in your responses.
            """
        
        # Create streaming chain
        rag_chain = (
            {
                "context": RunnablePassthrough(), 
                "question": RunnablePassthrough(),
                "conversation_context_section": lambda x: conversation_context_section
            }
            | self.prompt_template
            | self.llm
        )
        
        def response_generator():
            """Generator function that streams the response."""
            try:
                if stage_callback:
                    stage_callback("streaming", "Streame Antwort...")
                
                # Stream the response from the LLM
                for chunk in rag_chain.stream({"context": context, "question": query}):
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                    else:
                        content = str(chunk)
                    
                    if content:
                        yield content
                
                if stage_callback:
                    stage_callback("complete", "Antwort vollständig!")
                        
            except Exception as e:
                logging.error(f"Error during streaming: {e}")
                if stage_callback:
                    stage_callback("error", f"Fehler beim Streaming: {str(e)}")
                yield f"Fehler beim Generieren der Antwort: {str(e)}"
        
        return response_generator(), retrieved_docs, retrieved_metas 