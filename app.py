"""
Swiss Legal Chatbot - Main Application
A RAG-based chatbot for Swiss migration and asylum law.
"""
import streamlit as st

from config.settings import (APP_TITLE, DEV_MODE, ENABLE_REFLECTION,
                             OPENAI_API_KEY, ENABLE_STREAMING, ENABLE_STAGE_NOTIFICATIONS)
from rag.services import LegalChatbotService
from ui.components import (clear_footnotes, display_api_key_input,
                           display_chat_message, display_error_message,
                           display_response_with_footnotes, display_sources,
                           display_sources_sidebar, display_welcome_section,
                           display_stage_notification, create_streaming_response_container,
                           stream_response_chunks)
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
    
    # Clear cache if streaming method is not available (for development)
    if DEV_MODE:
        if st.sidebar.button("🔄 Cache leeren (Dev)"):
            st.cache_resource.clear()
            st.rerun()
    
    service = get_chatbot_service()
    
    # Verify streaming method is available
    if not hasattr(service, 'process_query_streaming'):
        st.error("⚠️ Streaming functionality not available. Please clear cache and restart.")
        if st.button("🔄 Cache leeren und neu laden"):
            st.cache_resource.clear()
            st.rerun()
        st.stop()
    
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
        
        # Generate assistant response
        with st.chat_message("assistant"):
            try:
                # Initialize variables
                response = ""
                sources = None
                reflection_info = None
                
                # Get streaming settings from session state (can be overridden by user)
                use_streaming = st.session_state.get('enable_streaming', ENABLE_STREAMING)
                use_stage_notifications = st.session_state.get('enable_stage_notifications', ENABLE_STAGE_NOTIFICATIONS)
                
                if use_streaming:
                    # Use streaming response with stage notifications
                    stage_container = None
                    current_stage_container = None
                    
                    def stage_callback(stage_key, stage_message):
                        nonlocal current_stage_container
                        if use_stage_notifications:
                            if current_stage_container:
                                # Clear previous stage notification
                                current_stage_container.empty()
                            current_stage_container = display_stage_notification(stage_key, stage_message)
                    
                    # Get streaming response
                    response_generator, retrieved_docs, retrieved_metas = service.process_query_streaming(
                        prompt, stage_callback=stage_callback if use_stage_notifications else None
                    )
                    
                    # Create response container for streaming
                    response_container = create_streaming_response_container()
                    
                    # Stream the response
                    response = stream_response_chunks(response_generator, response_container)
                    
                    # Clear final stage notification
                    if current_stage_container and use_stage_notifications:
                        current_stage_container.empty()
                    
                    # Create sources structure
                    sources = {"docs": retrieved_docs, "metas": retrieved_metas} if retrieved_docs else None
                    reflection_info = None  # Reflection not supported in streaming mode yet
                    
                    # Use the new footnote system for sources
                    if sources and sources.get("docs"):
                        # Clear the response container and redisplay with footnotes
                        response_container.empty()
                        footnotes = display_response_with_footnotes(response, sources)
                        # Store footnotes for sidebar display
                        if "current_footnotes" not in st.session_state:
                            st.session_state.current_footnotes = []
                        st.session_state.current_footnotes.extend(footnotes)
                    
                else:
                    # Use traditional non-streaming response
                    with st.spinner("Antwort wird gesucht und generiert..."):
                        result = service.process_query(prompt)
                        response = result["response"]
                        sources = result.get("sources")
                        reflection_info = result.get("reflection_info")

                        # Use the new footnote system
                        if sources:
                            footnotes = display_response_with_footnotes(response, sources)
                            # Store footnotes for sidebar display
                            if "current_footnotes" not in st.session_state:
                                st.session_state.current_footnotes = []
                            st.session_state.current_footnotes.extend(footnotes)
                        else:
                            st.markdown(response)
                        
                        # Display reflection information if available
                        if DEV_MODE:
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
    
    # Setup sidebar with footnotes
    setup_sidebar(st.session_state.get("messages", []), service.get_db_manager())


if __name__ == "__main__":
    main() 