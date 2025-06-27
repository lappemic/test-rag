"""
Swiss Legal Chatbot - Main Application
A RAG-based chatbot for Swiss migration and asylum law.
"""
import streamlit as st
from utils.logging_config import setup_logging
from config.settings import APP_TITLE, OPENAI_API_KEY, ENABLE_REFLECTION
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


def display_reflection_info(reflection_info):
    """Display reflection information in the UI."""
    if not reflection_info:
        return
    
    if reflection_info.get("error"):
        st.warning(f"🔄 Reflection failed: {reflection_info['error']}")
        return
    
    status = reflection_info.get("final_status", "unknown")
    iterations = reflection_info.get("iterations", 0)
    additional_sources = reflection_info.get("additional_sources_found", 0)
    
    if status == "no_reflection_needed":
        st.success("✅ Response was complete on first try")
    elif status == "completed_successfully":
        if iterations > 0:
            st.success(f"🔄 Response improved through {iterations} reflection iteration(s), found {additional_sources} additional source(s)")
    elif status == "max_iterations_reached":
        st.warning(f"🔄 Maximum reflection iterations ({iterations}) reached")
    elif status == "no_additional_sources_found":
        st.info(f"🔄 Reflection completed after {iterations} iteration(s) - no additional sources found")


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
    
    # Display reflection status if enabled
    if ENABLE_REFLECTION and service.is_reflection_enabled():
        st.info("🔄 Reflection pattern is enabled - responses will be automatically improved through iterative refinement")
    
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
                    reflection_info = result.get("reflection_info")

                    st.markdown(response)
                    
                    if sources:
                        display_sources(sources)
                    
                    # Display reflection information if available
                    if reflection_info and ENABLE_REFLECTION:
                        with st.expander("🔄 Reflection Details", expanded=False):
                            display_reflection_info(reflection_info)

                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                        "reflection_info": reflection_info
                    })
                    
                except Exception as e:
                    display_error_message(e)
    
    # Setup sidebar
    setup_sidebar(st.session_state.get("messages", []), service.get_db_manager())


if __name__ == "__main__":
    main() 