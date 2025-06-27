"""
Reusable UI components for the Swiss Legal Chatbot.
"""
import streamlit as st
from datetime import datetime


def display_welcome_section(loaded_laws):
    """Display the welcome section with available laws."""
    with st.expander("Willkommen! Was kann dieser Chatbot?", expanded=True):
        st.markdown("""
        Mit diesem Chatbot kannst du Fragen zu wichtigen Gesetzen für Migrations- und Asylbereiche in der Schweiz stellen. Der Bot hilft dir zum Beispiel bei Themen wie Aufenthaltsberechtigung, Arbeitssuche, Rechte, Pflichten und nächsten Schritten.

        **Stelle einfach deine Frage – der Bot sucht die passende Antwort in den Gesetzen.**
        
        ### 🧠 Konversationsgedächtnis
        Der Chatbot merkt sich eure Unterhaltung und kann auf vorherige Fragen und Antworten Bezug nehmen. 
        Stelle Folgefragen wie "Kannst du das genauer erklären?" oder "Was gilt in diesem Fall noch?"
        
        **Funktionen:**
        - 💬 Kontextbewusste Antworten
        - 🔄 Konversation zurücksetzen (Sidebar)
        - 📝 Unterhaltung exportieren (Sidebar)
        """)

        if not loaded_laws:
            st.warning("No law collections found in the database. Please run the ingestion script to load the laws.")
            st.code("bash ingest.sh", language="bash")
            return False
        else:
            st.markdown("**Aktuell kennt der Bot diese Gesetze:**")
            for law_name in loaded_laws:
                st.markdown(f"- {law_name}")
            return True


def display_chat_message(message):
    """Display a single chat message."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            display_sources(message["sources"])


def display_sources(sources):
    """Display sources in an expandable section."""
    if sources and sources.get("docs"):
        with st.expander("Quellen anzeigen"):
            for doc, meta in zip(sources["docs"], sources["metas"]):
                doc_title = meta.get('document_title', 'Unknown')
                art_id = meta.get('article_id', 'N/A')
                st.markdown(f"**Quelle:** {doc_title} - `{art_id}`")
                st.markdown(f"> {doc[:300]}...")
                st.divider()


def display_error_message(error):
    """Display an error message."""
    st.error(f"Error generating response: {error}")


def display_api_key_input():
    """Display API key input section."""
    st.text_input("OpenAI API Key", type="password", key="api_key_input")
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")


def export_conversation_button(messages):
    """Create export conversation functionality."""
    if not messages:
        return
        
    if st.sidebar.button("📝 Konversation exportieren"):
        conversation_text = []
        for msg in messages:
            role = "Sie" if msg["role"] == "user" else "Rechts-Chatbot"
            conversation_text.append(f"**{role}:** {msg['content']}\n")
        
        export_text = "\n".join(conversation_text)
        st.sidebar.download_button(
            label="📄 Als Textdatei herunterladen",
            data=export_text,
            file_name=f"rechts_chatbot_konversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        ) 