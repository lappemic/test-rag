"""
Document retrieval and RAG processing for the Swiss Legal Chatbot.
"""
import logging
import operator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from config.settings import MAX_RESULTS


class RAGRetriever:
    """Handles document retrieval and RAG processing."""
    
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
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Set up the RAG prompt template."""
        self.prompt_template = ChatPromptTemplate.from_template(
            """
You are a Swiss legal expert chatbot with comprehensive knowledge of Swiss federal laws, regulations, and legal principles, specializing in the Systematic Recompilation (SR) of Swiss legislation. Your task is to provide a precise, well-reasoned, and legally accurate answer to the user's question based on the provided context, which includes relevant laws, articles, paragraphs, or metadata from Swiss legal documents.

{conversation_context_section}

Adhere to the following guidelines:

1. **Conversation Awareness**:
   - Consider the conversation history when formulating your response
   - Reference previous topics discussed when relevant
   - Provide continuity in your responses while maintaining legal accuracy
   - Handle follow-up questions that may reference earlier parts of the conversation

2. **Accuracy and Relevance**:
   - Base your response exclusively on the provided context, ensuring all legal references are drawn from the retrieved chunks.
   - Cite specific sources using the format: `[Document Title, SR Number, Article ID/Paragraph ID, Date of Applicability]`, e.g., `[Asylgesetz (AsylG), SR 142.20, Art. 3/Para. 1, 2025-01-01]`.
   - If multiple chunks are relevant, prioritize the most specific (e.g., paragraph-level over article-level) and recent provisions based on `date_applicability`.

3. **Clarity and Structure**:
   - Organize your response in a logical, concise manner, using bullet points, numbered lists, or headings where appropriate to enhance readability.
   - Break down complex legal concepts into clear, understandable terms for users without legal expertise, while maintaining precision for expert users.
   - Avoid jargon unless necessary, and define any technical terms (e.g., "SR Number" as Systematic Recompilation Number).

4. **Direct Citations**:
   - For every legal point, explicitly reference the source chunk using the metadata fields (e.g., `document_title`, `sr_number`, `article_id`, `paragraph_id`).
   - Include the `date_applicability` to clarify the temporal validity of the cited provision.
   - If the chunk includes `references` or `amendment_history`, incorporate these to provide context for cross-referenced laws or changes in legislation.

5. **Comprehensive Analysis**:
   - Apply the cited provisions to the question or scenario, explaining how the law addresses the issue.
   - Address potential ambiguities, exceptions, or conflicting provisions within the context, referencing relevant `keywords` or `references` if available.
   - If the question involves interpretation, provide a reasoned analysis grounded in the legal text and metadata (e.g., `article_title` or `chapter_id` for thematic context).

6. **Neutral and Professional Tone**:
   - Maintain an objective, impartial tone, avoiding speculative or unsupported statements.
   - Do not assume facts or scenarios beyond the provided context or question.

7. **Handling Insufficient Context**:
   - If the provided context is insufficient to fully answer the question, clearly state: "The provided context does not contain sufficient information to fully address the question."
   - Offer a general legal principle or framework from Swiss law, if applicable, and note the limitation, e.g., "Based on general principles of Swiss federal law, [explain principle], but specific provisions require further context."
   - Suggest potential sources (e.g., "Relevant provisions may be found in [Document Title, SR Number]"). 

8. **Metadata Utilization**:
   - Leverage metadata fields (e.g., `document_title`, `sr_number`, `article_title`, `chapter_id`, `date_entry_in_force`) to provide context and ensure relevance.
   - Use `keywords` to identify related topics or filter relevant chunks.
   - Reference `amendment_history` to clarify whether a provision has been modified or superseded.

9. **Response to Queries About Known Laws**:
   - If asked about the laws you have knowledge of (e.g., "Welche Gesetze kennst du?"), list all unique `document_title` values from the context, along with their corresponding `sr_number`, in a clear and concise format, e.g., "Ich kenne folgende Gesetze: [Asylgesetz (AsylG), SR 142.20], [Schweizerische Bundesverfassung (BV), SR 101]."
   - Do not list specific articles or provisions unless explicitly requested by the user.
   - If the context contains no relevant `document_title` values, state: "No specific laws are available in the provided context."

10. **Error Handling**:
    - If the context contains contradictory or outdated chunks (based on `date_applicability`), prioritize the most recent and specific provision and note the discrepancy, e.g., "An earlier provision [cite] was superseded by [cite]."
    - If no relevant chunks are retrieved, state: "No relevant legal provisions were found in the provided context for this question."

11. **Language and Localization**:
    - Respond in the language of the context (`language` field, e.g., "de" for German) unless the user specifies otherwise.
    - If the user requests a response in another language (e.g., English), translate legal terms accurately and note: "This response is provided in English for clarity, but the original legal text is in [language]."

12. **Legal Disclaimer**:
    - Include in every response: "Diese Informationen dienen nur zur allgemeinen Orientierung und stellen keine Rechtsberatung dar. Für konkrete Fälle konsultieren Sie bitte einen qualifizierten Stelle."

LEGAL CONTEXT:
{context}

CURRENT QUESTION: {question}
"""
        )
    
    def query_rag(self, query, collections, num_results=None, conversation_history=None):
        """
        Query the RAG system with conversation memory.
        
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
        
        all_results = []
        for collection in collections:
            results = collection.query(
                query_embeddings=[query_embedding], 
                n_results=num_results,
                include=["metadatas", "documents", "distances"]
            )
            # Each result batch is for a single query embedding
            for i in range(len(results["ids"][0])):
                all_results.append({
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i]
                })

        # Sort all results by distance (ascending)
        sorted_results = sorted(all_results, key=operator.itemgetter('distance'))
        
        # Take the top N results overall
        top_results = sorted_results[:num_results]

        retrieved_docs = [res["document"] for res in top_results]
        retrieved_metas = [res["metadata"] for res in top_results]
        
        context = "\n".join([
            f"[Source: {meta.get('document_title', 'Unknown')} - {meta.get('article_id', 'Unknown') or 'meta'}]\n{doc}"
            for doc, meta in zip(retrieved_docs, retrieved_metas)
        ])
        
        logging.info(f"Retrieved {len(retrieved_docs)} documents for query across {len(collections)} collections.")
        
        # Format conversation context for the prompt
        conversation_context_section = ""
        if conversation_context:
            conversation_context_section = f"""
            CONVERSATION HISTORY:
            {conversation_context}

            Please consider this conversation history when answering the current question. Reference previous topics when relevant and maintain continuity in your responses.
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