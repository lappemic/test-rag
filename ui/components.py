"""
Reusable UI components for the Swiss Legal Chatbot.
"""
import re
from datetime import datetime

import streamlit as st

from config.settings import ENABLE_REFLECTION


def display_welcome_section(loaded_laws):
    """Display the welcome section with available laws."""
    with st.expander("Willkommen! Was kann dieser Chatbot?", expanded=True):
        st.markdown("""
        Mit diesem Chatbot kannst du Fragen zu wichtigen Gesetzen für Migrations- und Asylbereiche in der Schweiz stellen. Der Bot hilft dir zum Beispiel bei Themen wie Aufenthaltsberechtigung, Arbeitssuche, Rechte, Pflichten und nächsten Schritten.

        **Stelle einfach deine Frage – der Bot sucht die passende Antwort in den Gesetzen.**
        
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


def display_reflection_info_compact(reflection_info):
    """Display compact reflection information for historical messages."""
    if not reflection_info or not ENABLE_REFLECTION:
        return
    
    if reflection_info.get("error"):
        st.caption("🔄 Reflection failed")
        return
    
    status = reflection_info.get("final_status", "unknown")
    iterations = reflection_info.get("iterations", 0)
    additional_sources = reflection_info.get("additional_sources_found", 0)
    
    if status == "no_reflection_needed":
        st.caption("✅ Response complete on first try")
    elif status == "completed_successfully" and iterations > 0:
        st.caption(f"🔄 Improved through {iterations} reflection iteration(s), {additional_sources} additional source(s)")
    elif status == "max_iterations_reached":
        st.caption(f"🔄 Reflection completed ({iterations} iterations)")
    elif status == "no_additional_sources_found" and iterations > 0:
        st.caption(f"🔄 Reflection completed ({iterations} iterations)")


def add_footnote_markers(text, sources):
    """Add superscript footnote markers to the response text based on source relevance."""
    if not sources or not sources.get("docs"):
        return text, []
    
    # Keywords that suggest legal concepts worth footnoting
    legal_keywords = [
        "artikel", "art.", "absatz", "paragraph", "bestimmung", "vorschrift",
        "gesetz", "verordnung", "recht", "pflicht", "berechtigung", 
        "aufenthalt", "niederlassung", "erwerbstätigkeit", "arbeiten",
        "asyl", "schutz", "rückkehr", "ausschaffung", "wegweisung",
        "behörde", "entscheid", "verfügung", "beschwerde", "rekurs"
    ]
    
    footnotes = []
    modified_text = text
    footnote_positions = []  # Track where we've added footnotes to avoid overlap
    unfootnoted_sources = []  # Track sources that didn't get footnoted
    
    # Create footnotes for each source
    for i, (doc, meta) in enumerate(zip(sources["docs"], sources["metas"]), 1):
        doc_title = meta.get('document_title', 'Unknown Document')
        art_id = meta.get('article_id', 'N/A')
        
        footnotes.append({
            "id": i,
            "title": doc_title,
            "article": art_id,
            "content": doc,
            "meta": meta
        })
        
        # Find the most relevant position to place this footnote
        best_match = None
        best_relevance = 0
        
        # Check each legal keyword for relevance
        for keyword in legal_keywords:
            # Case-insensitive search for keywords
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(modified_text))
            
            # Calculate relevance score based on keyword frequency in source
            keyword_in_doc = doc.lower().count(keyword.lower())
            
            if matches and keyword_in_doc > 0:
                for match in matches:
                    # Check if this position is too close to existing footnotes
                    too_close = any(abs(match.start() - pos) < 20 for pos in footnote_positions)
                    if not too_close:
                        relevance = keyword_in_doc * (1.0 / (matches.index(match) + 1))  # Prefer earlier occurrences
                        if relevance > best_relevance:
                            best_match = match
                            best_relevance = relevance
        
        # Add footnote at the best position
        if best_match:
            pos = best_match.end()
            footnote_marker = f'<sup><a href="#footnote-{i}" style="color: #ff6b6b; text-decoration: none; font-weight: bold; background: rgba(255,107,107,0.1); padding: 1px 3px; border-radius: 3px; font-size: 0.75em;">[{i}]</a></sup>'
            modified_text = modified_text[:pos] + footnote_marker + modified_text[pos:]
            footnote_positions.append(pos)
            
            # Adjust positions for subsequent footnotes
            adjustment = len(footnote_marker)
            footnote_positions = [p + adjustment if p > pos else p for p in footnote_positions]
        else:
            unfootnoted_sources.append(i)
    
    # Fallback strategy: Add footnotes for sources that didn't get placed
    if unfootnoted_sources:
        # For simplicity, add remaining footnotes at the end of the first paragraph or sentence
        first_sentence_end = re.search(r'[.!?]\s+', modified_text)
        if first_sentence_end:
            pos = first_sentence_end.start() + 1
            for source_id in unfootnoted_sources:
                footnote_marker = f'<sup><a href="#footnote-{source_id}" style="color: #ff6b6b; text-decoration: none; font-weight: bold; background: rgba(255,107,107,0.1); padding: 1px 3px; border-radius: 3px; font-size: 0.75em;">[{source_id}]</a></sup>'
                modified_text = modified_text[:pos] + footnote_marker + modified_text[pos:]
                pos += len(footnote_marker)  # Adjust position for next footnote
        else:
            # If no sentence endings found, add at the end
            for source_id in unfootnoted_sources:
                footnote_marker = f'<sup><a href="#footnote-{source_id}" style="color: #ff6b6b; text-decoration: none; font-weight: bold; background: rgba(255,107,107,0.1); padding: 1px 3px; border-radius: 3px; font-size: 0.75em;">[{source_id}]</a></sup>'
                modified_text = modified_text + " " + footnote_marker
    
    return modified_text, footnotes


def display_response_with_footnotes(response_text, sources):
    """Display response text with inline footnote markers."""
    enhanced_text, footnotes = add_footnote_markers(response_text, sources)
    
    # Display the response with footnotes
    st.markdown(enhanced_text, unsafe_allow_html=True)
    
    return footnotes


def display_sources_sidebar(footnotes):
    """Display sources in the sidebar as a citation panel."""
    if not footnotes:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Quellen & Zitate")
    st.sidebar.markdown("*Die nummerierten Verweise im Text führen zu diesen Quellen*")
    st.sidebar.markdown("")
    
    for footnote in footnotes:
        # Create a more visually appealing citation card
        with st.sidebar.container():
            # Header with footnote number and title
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%); 
                        padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; 
                        border-left: 3px solid #ff6b6b;">
                <strong style="color: #ff6b6b;">[{footnote['id']}]</strong> 
                <span style="color: #262730; font-size: 14px;">{footnote['title']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Article information
            if footnote['article'] != 'N/A':
                st.markdown(f"**📖 Artikel:** `{footnote['article']}`")
            
            # Content preview with expand option
            with st.expander("📄 Volltext anzeigen", expanded=False):
                st.markdown(f"*{footnote['content']}*")
            
            # Add anchor for jumping
            st.markdown(f'<div id="footnote-{footnote["id"]}"></div>', unsafe_allow_html=True)
            
            # Small separator
            st.markdown("---")


def display_chat_message(message):
    """Display a single chat message with enhanced footnote system."""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # For assistant messages, use the enhanced footnote system
            if message.get("sources"):
                footnotes = display_response_with_footnotes(message["content"], message["sources"])
                
                # Store footnotes in session state for sidebar display
                if "current_footnotes" not in st.session_state:
                    st.session_state.current_footnotes = []
                st.session_state.current_footnotes.extend(footnotes)
            else:
                st.markdown(message["content"])
            
            # Display compact reflection info if available
            if message.get("reflection_info") and ENABLE_REFLECTION:
                display_reflection_info_compact(message["reflection_info"])
        else:
            # For user messages, display normally
            st.markdown(message["content"])


def display_sources(sources):
    """Legacy function - kept for backward compatibility but now shows inline footnotes."""
    # This function is now handled by display_response_with_footnotes
    # but we keep it to avoid breaking existing code
    pass


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


def clear_footnotes():
    """Clear stored footnotes from session state."""
    if "current_footnotes" in st.session_state:
        st.session_state.current_footnotes = [] 