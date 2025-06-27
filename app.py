"""
Swiss Legal Chatbot - Main Application
A RAG-based chatbot for Swiss migration and asylum law.
"""
import streamlit as st
from utils.logging_config import setup_logging
from config.settings import APP_TITLE, OPENAI_API_KEY
from rag.services import LegalChatbotService
from ui.components import (
    display_welcome_section, 
    display_chat_message, 
    display_sources,
    display_error_message,
    display_api_key_input
)
from ui.sidebar import setup_sidebar

# Initialize logging
setup_logging()

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide"
)

st.title(APP_TITLE)


def main():
    """Main application function."""
    # Check for API key
    if not OPENAI_API_KEY:
        display_api_key_input()
        st.stop()
    
    # Initialize the chatbot service
    @st.cache_resource
    def get_chatbot_service():
        return LegalChatbotService()
    
    service = get_chatbot_service()
    
    # Check if service is ready
    if not service.is_ready():
        st.error("Failed to initialize the chatbot service. Please check your API key and database collections.")
        st.stop()
    
    # Get loaded laws
    loaded_laws = service.get_loaded_laws()
    
    # Display welcome section
    if not display_welcome_section(loaded_laws):
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        display_chat_message(message)
    
    # Accept user input
    if prompt := st.chat_input("Stellen Sie eine Frage zu den Schweizer Gesetzen..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Antwort wird gesucht und generiert..."):
                try:
                    result = service.process_query(prompt)
                    response = result["response"]
                    sources = result.get("sources")

                    st.markdown(response)
                    
                    if sources:
                        display_sources(sources)

                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    display_error_message(e)
    
    # Setup sidebar
    setup_sidebar(st.session_state.get("messages", []), service.get_db_manager())


if __name__ == "__main__":
    main() 