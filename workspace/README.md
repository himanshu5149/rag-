# Local RAG Agent - Fully Browser-Based

A complete, self-contained Retrieval-Augmented Generation (RAG) application that runs entirely in your browser. No API keys, no external services, no backend required.

## Features

✅ **100% Local** - All processing happens in your browser  
✅ **Real LLM** - Uses Llama 3.2 1B via WebLLM (WebGPU)  
✅ **Real Embeddings** - Uses all-MiniLM-L6-v2 via Transformers.js  
✅ **PDF Support** - Extract text from PDF files  
✅ **Vector Search** - Cosine similarity search over document chunks  
✅ **No Setup** - Just open the app and start using it  

## How It Works

1. **First Load**: Downloads models (~2GB total, cached in browser)
   - Llama 3.2 1B for text generation
   - all-MiniLM-L6-v2 for embeddings

2. **Upload Documents**: Upload PDFs or paste text
   - Text is extracted and chunked
   - Each chunk is embedded and stored in vector database

3. **Ask Questions**: The RAG pipeline:
   - Embeds your question
   - Searches for relevant chunks (cosine similarity)
   - Sends context + question to LLM
   - Returns grounded answer with sources

## Requirements

- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- ~2GB disk space for model cache (one-time download)

## Usage

1. Open the app in your browser
2. Wait for models to load (progress bar shown)
3. Upload a PDF or paste text in the sidebar
4. Ask questions in the chat interface
5. View retrieved sources by clicking "X sources retrieved"

## Architecture

A fully browser-based RAG implementation:

- **PDF Parsing**: pdfjs-dist
- **Embeddings**: @xenova/transformers (all-MiniLM-L6-v2)
- **LLM**: @mlc-ai/web-llm (Llama 3.2 1B)
- **Vector Store**: Custom in-memory implementation with cosine similarity
- **Chunking**: Custom implementation with overlap

## Privacy

All data stays in your browser. Nothing is sent to external servers. Models are downloaded once from Hugging Face and cached locally.

## Limitations

- First load takes time (downloading ~2GB of models)
- Requires WebGPU support (not available in all browsers)
- LLM quality is limited by model size (1B parameters)
- Large PDFs may be slow to process

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

The built app is fully static and can be deployed anywhere.
