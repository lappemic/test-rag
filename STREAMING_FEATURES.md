# Streaming Features Documentation

## Overview

The Swiss Legal Chatbot now supports **answer streaming** with **real-time stage notifications**, providing users with immediate feedback about the processing status and streaming responses as they are generated.

## New Features

### 1. Answer Streaming
- **Real-time response generation**: See the answer being written as the AI generates it
- **Enhanced user experience**: No more waiting for complete responses
- **Configurable**: Can be toggled on/off via sidebar controls

### 2. Stage Notifications
- **Processing transparency**: Users see exactly which stage the system is currently processing
- **German language notifications**: All stage messages are in German for consistency
- **Visual indicators**: Each stage has its own emoji and color coding

### Stage Breakdown

The RAG system processes queries through these stages:

1. **🧠 Aufbau des Gesprächskontexts** - Building conversation context
2. **✨ Verbesserung der Anfrage** - Query enhancement  
3. **🔢 Erstelle Einbettungen** - Creating embeddings
4. **🔍 Filtere relevante Rechtssammlungen** - Filtering relevant law collections
5. **📚 Durchsuche X Rechtssammlungen** - Searching X law collections
6. **⚙️ Verarbeite gefundene Dokumente** - Processing found documents
7. **💭 Generiere Antwort** - Generating answer
8. **📝 Streame Antwort** - Streaming answer
9. **✅ Antwort vollständig** - Answer complete

## Configuration

### Settings (config/settings.py)
```python
# Streaming settings
ENABLE_STREAMING = True                  # Enable response streaming
STREAMING_CHUNK_SIZE = 25               # Number of words per streaming chunk  
ENABLE_STAGE_NOTIFICATIONS = True       # Show processing stage updates to user
```

## Technical Implementation

### Components Added

#### 1. RAGRetriever.query_rag_streaming()
New streaming method in `rag/retrieval.py` that:
- Supports stage callbacks for notifications
- Returns a generator for streaming responses
- Maintains compatibility with existing footnote system

#### 2. LegalChatbotService.process_query_streaming()
New service method in `rag/services.py` that:
- Handles streaming requests with stage callbacks
- Manages both simple queries (law lists) and complex RAG queries
- Provides error handling for streaming scenarios

#### 3. UI Components
New functions in `ui/components.py`:
- `display_stage_notification()`: Shows processing stage updates
- `create_streaming_response_container()`: Creates containers for streaming
- `stream_response_chunks()`: Handles real-time response display

#### 4. Enhanced Main App
Updated `app.py` to:
- Support both streaming and non-streaming modes
- Use session state for real-time settings changes
- Maintain full compatibility with footnotes and sources

### Streaming Flow

```
User Query → Stage Notifications → Document Retrieval → Response Streaming → Footnotes Display
```

## Benefits

1. **Improved UX**: Users get immediate feedback and see progress
2. **Transparency**: Clear visibility into processing stages
3. **Flexibility**: Users can choose between streaming and traditional modes
4. **Performance**: Perceived faster response times through streaming
5. **Compatibility**: Full compatibility with existing features (footnotes, sources, reflection)

## Usage Examples

### Traditional Mode
```python
result = service.process_query(query)
response = result["response"]
```

### Streaming Mode
```python
response_generator, docs, metas = service.process_query_streaming(
    query, 
    stage_callback=my_stage_callback
)
```

## Future Enhancements

- **Reflection + Streaming**: Integration of reflection pattern with streaming
- **Progress bars**: Visual progress indicators for each stage
- **Customizable stages**: User-configurable stage visibility
- **Performance metrics**: Display timing information for each stage

## Notes

- Reflection pattern is currently disabled in streaming mode (complexity considerations)
- Stage notifications are automatically cleared when moving to the next stage
- Streaming works seamlessly with the existing footnote and source citation system
- All error states are handled gracefully with appropriate user feedback 