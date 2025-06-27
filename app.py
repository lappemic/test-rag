"""
Swiss Legal Chatbot - Main Application
A RAG-based chatbot for Swiss migration and asylum law.
"""
import streamlit as st

from config.settings import (APP_TITLE, DEV_MODE, ENABLE_REFLECTION,
                             OPENAI_API_KEY)
from rag.services import LegalChatbotService
from ui.components import (clear_footnotes, display_api_key_input,
                           display_chat_message, display_error_message,
                           display_reflection_info_detailed, display_response_with_footnotes, 
                           display_sources, display_sources_sidebar, display_welcome_section)
from ui.sidebar import setup_sidebar
from utils.logging_config import setup_logging

# Initialize logging
setup_logging()

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for enhanced footnote styling
st.markdown("""
<style>
/* Footnote styling */
sup a {
    transition: all 0.2s ease-in-out;
    border-radius: 4px !important;
    margin-left: 2px;
}

sup a:hover {
    background: rgba(255,107,107,0.2) !important;
    transform: scale(1.1);
    box-shadow: 0 2px 4px rgba(255,107,107,0.3);
}

/* Sidebar styling improvements */
.css-1d391kg {
    background-color: #f8f9fa;
}

/* Citation card styling */
.citation-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Smooth scrolling for footnote navigation */
html {
    scroll-behavior: smooth;
}

/* Enhanced expander styling for sources */
.streamlit-expanderHeader {
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* Chat message styling improvements */
.stChatMessage {
    background-color: #ffffff;
    border-radius: 8px;
    margin-bottom: 16px;
}

/* Sidebar section dividers */
.sidebar-divider {
    border-top: 2px solid #ff6b6b;
    margin: 20px 0;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

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
    
    # Display reflection status if enabled
    if DEV_MODE:
        if ENABLE_REFLECTION and service.is_reflection_enabled():
            st.info("🔄 Reflection pattern is enabled - responses will be automatically improved through iterative refinement")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Clear footnotes for fresh conversation display
    clear_footnotes()
    
    # Display chat messages from history
    for message in st.session_state.messages:
        display_chat_message(message)
    
    # Accept user input
    if prompt := st.chat_input("Stellen Sie eine Frage zu den Schweizer Gesetzen..."):
        # Clear previous footnotes when starting new query
        clear_footnotes()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response with streaming
        with st.chat_message("assistant"):
            # Status display for real-time updates
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            reflection_placeholder = st.empty()
            
            try:
                response_chunks = []
                sources = None
                reflection_info = None
                
                # Status callback for real-time updates
                def update_status(status_text):
                    status_placeholder.info(status_text)
                
                # Stream the response
                for chunk, is_complete, metadata in service.process_query_stream(prompt, status_callback=update_status):
                    if not is_complete:
                        response_chunks.append(chunk)
                        # Update response display in real-time
                        current_response = "".join(response_chunks)
                        response_placeholder.markdown(current_response)
                        
                        # Display reflection iteration info if available
                        if metadata and metadata.get("reflection_info"):
                            refl_info = metadata["reflection_info"]
                            current_iter = refl_info.get("current_iteration", 0)
                            max_iter = refl_info.get("max_iterations", 0)
                            if current_iter > 0 and ENABLE_REFLECTION:
                                reflection_placeholder.info(f"🔄 Reflexions-Iteration {current_iter}/{max_iter} läuft...")
                    else:
                        # Processing complete
                        status_placeholder.empty()
                        if metadata:
                            sources = metadata.get("sources")
                            reflection_info = metadata.get("reflection_info")
                            final_response = metadata.get("final_response", "".join(response_chunks))
                        else:
                            final_response = "".join(response_chunks)
                
                # Clear temporary placeholders
                status_placeholder.empty()
                reflection_placeholder.empty()
                response_placeholder.empty()
                
                # Display final response with footnotes
                if sources:
                    footnotes = display_response_with_footnotes(final_response, sources)
                    # Store footnotes for sidebar display
                    if "current_footnotes" not in st.session_state:
                        st.session_state.current_footnotes = []
                    st.session_state.current_footnotes.extend(footnotes)
                else:
                    st.markdown(final_response)
                
                # Display reflection information if available
                if DEV_MODE and reflection_info and ENABLE_REFLECTION:
                    with st.expander("🔄 Reflection Details", expanded=False):
                        display_reflection_info_detailed(reflection_info)

                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "sources": sources,
                    "reflection_info": reflection_info
                })
                
            except Exception as e:
                status_placeholder.empty()
                reflection_placeholder.empty()
                response_placeholder.empty()
                display_error_message(e)
    
    # Setup sidebar with footnotes
    setup_sidebar(st.session_state.get("messages", []), service.get_db_manager())


if __name__ == "__main__":
    main() 