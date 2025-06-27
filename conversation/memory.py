"""
Conversation memory and context management for the Swiss Legal Chatbot.
"""
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config.settings import MAX_CONVERSATION_LENGTH, MAX_RECENT_MESSAGES


class ConversationManager:
    """Manages conversation history, context, and summarization."""
    
    def __init__(self, llm):
        """Initialize with language model for summarization."""
        self.llm = llm
    
    def summarize_conversation(self, messages, max_length=None):
        """
        Summarizes conversation history when it gets too long.
        
        Args:
            messages: List of message dictionaries
            max_length: Maximum character length before summarization
        
        Returns:
            Summarized conversation string or None if not needed
        """
        if max_length is None:
            max_length = MAX_CONVERSATION_LENGTH
            
        # Calculate total length of conversation
        total_length = sum(len(msg["content"]) for msg in messages)
        
        if total_length < max_length:
            return None
            
        # Take messages from the middle of conversation (skip very recent ones)
        messages_to_summarize = messages[:-4] if len(messages) > 4 else messages[:-2]
        
        conversation_text = []
        for msg in messages_to_summarize:
            role = "Benutzer" if msg["role"] == "user" else "Assistent"
            conversation_text.append(f"{role}: {msg['content']}")
        
        summary_prompt = ChatPromptTemplate.from_template(
            """Fasse die folgende Konversation zwischen einem Benutzer und einem Rechts-Chatbot zusammen. 
            Konzentriere dich auf die wichtigsten rechtlichen Themen, Fragen und Antworten.
            
            KONVERSATION:
            {conversation}
            
            Antworte mit einer prägnanten Zusammenfassung (max. 300 Wörter):"""
        )
        
        summary_chain = summary_prompt | self.llm | StrOutputParser()
        
        try:
            summary = summary_chain.invoke({"conversation": "\n".join(conversation_text)})
            return f"[Zusammenfassung der bisherigen Konversation: {summary}]"
        except Exception as e:
            logging.warning(f"Conversation summarization failed: {e}")
            return None
    
    def build_conversation_context(self, messages, max_messages=None):
        """
        Builds conversation context from recent messages, with automatic summarization for long conversations.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_messages: Maximum number of recent messages to include
        
        Returns:
            String containing formatted conversation context
        """
        if max_messages is None:
            max_messages = MAX_RECENT_MESSAGES
            
        if not messages or len(messages) < 2:
            return ""
        
        context_parts = []
        
        # Check if conversation needs summarization
        if len(messages) > 8:
            summary = self.summarize_conversation(messages)
            if summary:
                context_parts.append(summary)
                # Use only the most recent messages after summarization
                max_messages = 2
        
        # Get recent messages (excluding the current one being processed)
        recent_messages = messages[-max_messages-1:-1] if len(messages) > max_messages else messages[:-1]
        
        for msg in recent_messages:
            role = "Benutzer" if msg["role"] == "user" else "Assistent"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts) if context_parts else "" 