import logging
import operator
import os
from datetime import datetime

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# dirty dev hack
DEV_MODE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("Schwizerischer Rechts-Chatbot im Migrationsrecht")

# --- Helper functions ---

def get_law_collections():
    """Gets all collections from ChromaDB that start with 'law_'."""
    try:
        collections = chroma_client.list_collections()
        law_collections = [c for c in collections if c.name.startswith("law_")]
        logging.info(f"Found {len(law_collections)} law collections.")
        return law_collections
    except Exception as e:
        logging.error(f"Error getting law collections: {e}")
        st.error("Could not connect to ChromaDB or find collections. Please run the ingestion script first.")
        return []

def get_loaded_law_names(collections):
    """Extracts document titles from the metadata of each collection."""
    law_names = set()
    for collection in collections:
        try:
            # Get one item to find the document title. We assume it's consistent.
            meta = collection.get(limit=1, include=["metadatas"])
            if meta["metadatas"]:
                doc_title = meta["metadatas"][0].get("document_title")
                if doc_title:
                    law_names.add(doc_title)
        except Exception as e:
            logging.error(f"Error getting metadata from collection {collection.name}: {e}")
    return sorted(list(law_names))


# --- Main Application ---

if not openai_api_key:
    st.text_input("OpenAI API Key", type="password", key="api_key_input")
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
    st.stop()

# --- Initialize models and DB connection ---
try:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")
    law_collections = get_law_collections()
except Exception as e:
    st.error(f"Failed to initialize OpenAI models. Please check your API key. Error: {e}")
    st.stop()


# --- Display Introduction and Available Laws ---
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

    if not law_collections:
        st.warning("No law collections found in the database. Please run the ingestion script to load the laws.")
        st.code("bash ingest.sh", language="bash")
        st.stop()
    else:
        st.markdown("**Aktuell kennt der Bot diese Gesetze:**")
        loaded_laws = get_loaded_law_names(law_collections)
        for law_name in loaded_laws:
            st.markdown(f"- {law_name}")


# Function to summarize long conversation history
def summarize_conversation(messages, llm, max_length=2000):
    """
    Summarizes conversation history when it gets too long.
    
    Args:
        messages: List of message dictionaries
        llm: Language model for summarization
        max_length: Maximum character length before summarization
    
    Returns:
        Summarized conversation string
    """
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
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    try:
        summary = summary_chain.invoke({"conversation": "\n".join(conversation_text)})
        return f"[Zusammenfassung der bisherigen Konversation: {summary}]"
    except Exception as e:
        logging.warning(f"Conversation summarization failed: {e}")
        return None


# Function to build conversation context from recent messages
def build_conversation_context(messages, max_messages=4, llm=None):
    """
    Builds conversation context from recent messages, with automatic summarization for long conversations.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_messages: Maximum number of recent messages to include
        llm: Language model for summarization (optional)
    
    Returns:
        String containing formatted conversation context
    """
    if not messages or len(messages) < 2:
        return ""
    
    context_parts = []
    
    # Check if conversation needs summarization
    if llm and len(messages) > 8:
        summary = summarize_conversation(messages, llm)
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


def enhance_query_with_context(current_query, conversation_context, llm):
    """
    Enhances the current query with conversation context to improve retrieval.
    
    Args:
        current_query: The user's current question
        conversation_context: Recent conversation history
        llm: Language model for query enhancement
    
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
    
    enhancement_chain = enhancement_prompt | llm | StrOutputParser()
    
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


# Function to query the RAG system with conversation memory
def query_rag(query, collections, llm, num_results, embeddings, conversation_history=None):
    logging.info(f"Querying RAG system with query: {query}")
    
    # Build conversation context
    conversation_context = build_conversation_context(conversation_history or [], llm=llm)
    
    # Enhance query with conversation context for better retrieval
    enhanced_query = enhance_query_with_context(query, conversation_context, llm)
    
    # Use enhanced query for embedding and retrieval
    query_embedding = embeddings.embed_query(enhanced_query)
    
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
    
    # Updated prompt template that includes conversation context
    prompt_template = ChatPromptTemplate.from_template(
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
        | prompt_template
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke({"context": context, "question": query})
    logging.info("Generated response from RAG chain with conversation context")
    return response, retrieved_docs, retrieved_metas


# --- Router and Chains ---

# 1. Router to decide where to send the query
routing_prompt_template = ChatPromptTemplate.from_template(
    """You are an expert at routing a user question to a vectorstore or a special function.

The vectorstore contains documents about Swiss law and can answer specific legal questions.

The special function 'list_known_laws' can list all the laws the chatbot knows about. It should be used for questions like "What laws do you know?", "Welche Gesetze kannst du beantworten?", or "Welche Gesetze kennst du?".

For which of these two options is the user question asking for?

Respond with a single word: "vectorstore" or "list_known_laws".

User Question: {question}"""
)

router = routing_prompt_template | llm | StrOutputParser()

# 2. Chain for listing known laws
def list_laws_chain(_: dict) -> dict:
    """
    A chain that gets the loaded law names, formats them, and returns them
    in a dictionary. The input is ignored.
    """
    loaded_laws = get_loaded_law_names(law_collections)
    if not loaded_laws:
        response_text = "I currently don't have any specific laws loaded in my database. Please run the ingestion script."
    else:
        law_list = "\n".join([f"- {law_name}" for law_name in loaded_laws])
        response_text = f"Ich kenne folgende Gesetze:\n\n{law_list}"
    
    disclaimer = '\n\nDiese Informationen dienen nur zur allgemeinen Orientierung und stellen keine Rechtsberatung dar. Für konkrete Fälle konsultieren Sie bitte einen qualifizierten Stelle.'
    
    return {
        "response": response_text + disclaimer,
        "sources": {"docs": [], "metas": []}
    }


# 3. RAG Chain wrapped in a function for the router
def rag_chain_func(input_dict: dict) -> dict:
    """
    Takes an input dictionary with a 'question' key and runs the full RAG pipeline.
    """
    query = input_dict["question"]
    # Get conversation history from session state
    conversation_history = st.session_state.get("messages", [])
    response, retrieved_docs, retrieved_metas = query_rag(
        query, law_collections, llm, num_results=5, embeddings=embeddings, 
        conversation_history=conversation_history
    )
    return {
        "response": response,
        "sources": {"docs": retrieved_docs, "metas": retrieved_metas}
    }

# 4. The full chain with branching
branch = RunnableBranch(
    (lambda x: "list_known_laws" in x["topic"].lower(), list_laws_chain),
    rag_chain_func
)

chain = {"topic": router, "question": lambda x: x["question"]} | branch


# --- Chat Interface ---

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            sources = message.get("sources", {})
            if sources.get("docs"):
                with st.expander("Quellen anzeigen"):
                    for doc, meta in zip(sources["docs"], sources["metas"]):
                        doc_title = meta.get('document_title', 'Unknown')
                        art_id = meta.get('article_id', 'N/A')
                        st.markdown(f"**Quelle:** {doc_title} - `{art_id}`")
                        st.markdown(f"> {doc[:300]}...")
                        st.divider()

# Accept user input
if prompt := st.chat_input("Stellen Sie eine Frage zu den Schweizer Gesetzen..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate assistant response with RAG
    with st.chat_message("assistant"):
        with st.spinner("Antwort wird gesucht und generiert..."):
            try:
                result = chain.invoke({"question": prompt})
                response = result["response"]
                sources = result.get("sources")

                st.markdown(response)

                if sources and sources.get("docs"):
                    with st.expander("Quellen anzeigen"):
                        for doc, meta in zip(sources["docs"], sources["metas"]):
                            doc_title = meta.get('document_title', 'Unknown')
                            art_id = meta.get('article_id', 'N/A')
                            st.markdown(f"**Quelle:** {doc_title} - `{art_id}`")
                            st.markdown(f"> {doc[:300]}...")
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
            except Exception as e:
                logging.error(f"Error generating response: {e}")
                st.error(f"Error generating response: {e}")

# --- Sidebar Controls ---
st.sidebar.title("Konversation")

# Conversation reset button
if st.sidebar.button("🔄 Neue Konversation starten"):
    st.session_state.messages = []
    st.sidebar.success("Konversation zurückgesetzt!")
    st.rerun()

# Show conversation stats and export option
if "messages" in st.session_state and st.session_state.messages:
    num_messages = len(st.session_state.messages)
    st.sidebar.metric("Nachrichten in dieser Konversation", num_messages)
    
    # Export conversation
    if st.sidebar.button("📝 Konversation exportieren"):
        conversation_text = []
        for msg in st.session_state.messages:
            role = "Sie" if msg["role"] == "user" else "Rechts-Chatbot"
            conversation_text.append(f"**{role}:** {msg['content']}\n")
        
        export_text = "\n".join(conversation_text)
        st.sidebar.download_button(
            label="📄 Als Textdatei herunterladen",
            data=export_text,
            file_name=f"rechts_chatbot_konversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if DEV_MODE:
    st.sidebar.title("Entwickler-Einstellungen")
    if st.sidebar.button("Alle Gesetze aus der Datenbank löschen"):
        if law_collections:
            try:
                with st.spinner("Lösche alle Gesetzessammlungen..."):
                    for collection in law_collections:
                        chroma_client.delete_collection(name=collection.name)
                st.sidebar.success("Alle Gesetzessammlungen wurden gelöscht. Bitte laden Sie die Seite neu.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Fehler beim Löschen der Sammlungen: {e}")
        else:
            st.sidebar.warning("Keine Sammlungen zum Löschen gefunden.")
