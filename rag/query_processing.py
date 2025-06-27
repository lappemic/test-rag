"""
Query processing and enhancement for the Swiss Legal Chatbot RAG system.
"""
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class QueryProcessor:
    """Processes and enhances queries for better retrieval."""
    
    def __init__(self, llm):
        """Initialize with language model for query enhancement."""
        self.llm = llm
    
    def enhance_query_with_context(self, current_query, conversation_context):
        """
        Enhances the current query with conversation context to improve retrieval.
        
        Args:
            current_query: The user's current question
            conversation_context: Recent conversation history
        
        Returns:
            Enhanced query string that's more self-contained
        """
        if not conversation_context:
            return current_query
        
        enhancement_prompt = ChatPromptTemplate.from_template(
            """Du bist ein Experte darin, Benutzeranfragen zu verbessern. Gegeben ist eine aktuelle Frage und der Konversationskontext.

            Deine Aufgabe ist es, die aktuelle Frage zu einer eigenständigen, vollständigen Suchanfrage umzuformulieren, die:
            1. Den Konversationskontext berücksichtigt
            2. Alle notwendigen Informationen enthält
            3. Für die Suche in rechtlichen Dokumenten optimiert ist
            4. Referenzen auf vorherige Antworten ("das", "diese", "wie oben erwähnt") auflöst

            KONVERSATIONSKONTEXT:
            {conversation_context}

            AKTUELLE FRAGE: {current_query}

            Antworte nur mit der verbesserten Suchanfrage, ohne zusätzliche Erklärungen:"""
        )
        
        enhancement_chain = enhancement_prompt | self.llm | StrOutputParser()
        
        try:
            enhanced_query = enhancement_chain.invoke({
                "conversation_context": conversation_context,
                "current_query": current_query
            })
            logging.info(f"Enhanced query: {current_query} -> {enhanced_query}")
            return enhanced_query.strip()
        except Exception as e:
            logging.warning(f"Query enhancement failed: {e}. Using original query.")
            return current_query 