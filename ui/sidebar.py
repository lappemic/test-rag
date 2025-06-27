"""
Sidebar functionality for the Swiss Legal Chatbot.
"""
import streamlit as st

from config.settings import DEV_MODE, ENABLE_STREAMING, ENABLE_STAGE_NOTIFICATIONS
from ui.components import display_sources_sidebar, export_conversation_button


def setup_sidebar(messages, db_manager):
    """Set up the complete sidebar with all controls."""
    st.sidebar.title("Konversation")
    
    # Conversation reset button
    if st.sidebar.button("🔄 Neue Konversation starten"):
        st.session_state.messages = []
        if "current_footnotes" in st.session_state:
            st.session_state.current_footnotes = []
        st.sidebar.success("Konversation zurückgesetzt!")
        st.rerun()
    
    # Show conversation stats and export option
    if messages:
        num_messages = len(messages)
        st.sidebar.metric("Nachrichten in dieser Konversation", num_messages)
        export_conversation_button(messages)
    
    # Display footnotes/citations if available
    if hasattr(st.session_state, 'current_footnotes') and st.session_state.current_footnotes:
        display_sources_sidebar(st.session_state.current_footnotes)
    
    # Developer mode section
    if DEV_MODE:
        setup_dev_sidebar(db_manager)


def setup_dev_sidebar(db_manager):
    """Set up developer-specific sidebar options."""
    st.sidebar.title("Entwickler-Einstellungen")
    if st.sidebar.button("Alle Gesetze aus der Datenbank löschen"):
        law_collections = db_manager.get_law_collections()
        if law_collections:
            try:
                with st.spinner("Lösche alle Gesetzessammlungen..."):
                    num_deleted = db_manager.delete_all_law_collections()
                st.sidebar.success(f"Alle {num_deleted} Gesetzessammlungen wurden gelöscht. Bitte laden Sie die Seite neu.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Fehler beim Löschen der Sammlungen: {e}")
        else:
            st.sidebar.warning("Keine Sammlungen zum Löschen gefunden.") 