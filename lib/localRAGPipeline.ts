// Fully self-contained RAG pipeline using browser-based models
// Uses WebLLM for real LLM inference + Transformers.js for embeddings
// No external services - everything runs locally in the browser

import { CreateMLCEngine, MLCEngine } from '@mlc-ai/web-llm';
import { pipeline, env } from '@xenova/transformers';
import { extractTextFromPDF } from './pdfParser';
import { chunkText, chunkPages } from './chunker';
import { VectorStore, SearchResult } from './vectorStore';

// Configure transformers
env.allowLocalModels = false;
env.useBrowserCache = true;

export interface RAGConfig {
  embeddingModel: string;
  llmModel: string;
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
}

export const DEFAULT_CONFIG: RAGConfig = {
  embeddingModel: 'Xenova/all-MiniLM-L6-v2',
  llmModel: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
  chunkSize: 500,
  chunkOverlap: 50,
  topK: 3,
};

export class LocalRAGPipeline {
  private embeddingPipeline: any = null;
  private llmEngine: MLCEngine | null = null;
  private vectorStore: VectorStore;
  private config: RAGConfig;
  private processedSources: Set<string> = new Set();
  private isLoading: boolean = false;

  constructor(config: Partial<RAGConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.vectorStore = new VectorStore('local-rag-browser');
  }

  // Initialize models - downloads and loads them into browser memory
  async initialize(onProgress?: (stage: string, progress: number) => void): Promise<void> {
    if (this.isLoading) return;
    this.isLoading = true;

    try {
      // Load embedding model (small, fast)
      onProgress?.('Loading embedding model...', 0);
      this.embeddingPipeline = await pipeline(
        'feature-extraction',
        this.config.embeddingModel,
        {
          progress_callback: (progress: any) => {
            if (progress.progress) {
              onProgress?.(`Downloading embeddings model: ${(progress.progress * 100).toFixed(0)}%`, progress.progress * 100);
            }
          }
        }
      );

      // Load LLM (larger, takes longer)
      onProgress?.('Loading LLM model...', 0);
      this.llmEngine = await CreateMLCEngine(
        this.config.llmModel,
        {
          initProgressCallback: (progress: any) => {
            if (progress.progress) {
              onProgress?.(`Downloading LLM: ${(progress.progress * 100).toFixed(0)}%`, progress.progress * 100);
            }
          }
        }
      );

      onProgress?.('All models loaded successfully!', 100);
    } catch (error) {
      // Clean up on error
      this.llmEngine = null;
      this.embeddingPipeline = null;
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  // Check if engine is still valid
  private isEngineValid(): boolean {
    return this.llmEngine !== null;
  }

  // Reload the LLM engine if it was disposed
  async reloadEngine(onProgress?: (stage: string, progress: number) => void): Promise<void> {
    if (this.llmEngine) {
      try {
        await this.llmEngine.unload();
      } catch {
        // Ignore errors during unload
      }
      this.llmEngine = null;
    }

    onProgress?.('Reloading LLM model...', 0);
    this.llmEngine = await CreateMLCEngine(
      this.config.llmModel,
      {
        initProgressCallback: (progress: any) => {
          if (progress.progress) {
            onProgress?.(`Reloading LLM: ${(progress.progress * 100).toFixed(0)}%`, progress.progress * 100);
          }
        }
      }
    );
    onProgress?.('LLM reloaded successfully!', 100);
  }

  // Check if models are loaded
  isReady(): boolean {
    return this.embeddingPipeline !== null && this.llmEngine !== null;
  }

  // Generate embedding for a single text
  private async getEmbedding(text: string): Promise<number[]> {
    if (!this.embeddingPipeline) {
      throw new Error('Embedding model not loaded. Call initialize() first.');
    }

    const output = await this.embeddingPipeline(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
  }

  // Generate embeddings for multiple texts
  private async getEmbeddings(texts: string[], onProgress?: (current: number, total: number) => void): Promise<number[][]> {
    const embeddings: number[][] = [];
    for (let i = 0; i < texts.length; i++) {
      const embedding = await this.getEmbedding(texts[i]);
      embeddings.push(embedding);
      if (onProgress) onProgress(i + 1, texts.length);
    }
    return embeddings;
  }

  // Ingest a PDF file
  async ingestPDF(
    file: File,
    onProgress?: (stage: string, current: number, total: number) => void
  ): Promise<{ chunks: number; source: string }> {
    if (!this.isReady()) {
      throw new Error('Models not loaded. Please wait for initialization.');
    }

    const source = file.name;

    if (this.processedSources.has(source)) {
      return { chunks: 0, source };
    }

    // Extract text from PDF
    onProgress?.('Extracting text from PDF...', 0, 1);
    const pages = await extractTextFromPDF(file, (current, total) => {
      onProgress?.(`Extracting page ${current}/${total}...`, current, total);
    });

    if (pages.length === 0) {
      throw new Error('No text could be extracted from this PDF.');
    }

    // Chunk the text
    onProgress?.('Chunking text...', 0, 1);
    const chunks = chunkPages(pages, {
      chunkSize: this.config.chunkSize,
      chunkOverlap: this.config.chunkOverlap,
      source,
    });

    if (chunks.length === 0) {
      throw new Error('No chunks were generated.');
    }

    // Generate embeddings
    const texts = chunks.map((c) => c.text);
    const embeddings = await this.getEmbeddings(texts, (current, total) => {
      onProgress?.(`Embedding chunk ${current}/${total}...`, current, total);
    });

    // Store in vector database
    this.vectorStore.addDocuments(chunks, embeddings);
    this.processedSources.add(source);

    return { chunks: chunks.length, source };
  }

  // Ingest plain text
  async ingestText(
    text: string,
    source: string = 'pasted-text',
    onProgress?: (stage: string, current: number, total: number) => void
  ): Promise<{ chunks: number; source: string }> {
    if (!this.isReady()) {
      throw new Error('Models not loaded. Please wait for initialization.');
    }

    if (this.processedSources.has(source)) {
      return { chunks: 0, source };
    }

    onProgress?.('Chunking text...', 0, 1);
    const chunks = chunkText(text, {
      chunkSize: this.config.chunkSize,
      chunkOverlap: this.config.chunkOverlap,
      source,
    });

    if (chunks.length === 0) {
      throw new Error('No chunks were generated.');
    }

    const texts = chunks.map((c) => c.text);
    onProgress?.('Generating embeddings...', 0, 1);
    const embeddings = await this.getEmbeddings(texts, (current, total) => {
      onProgress?.(`Embedding chunk ${current}/${total}...`, current, total);
    });

    this.vectorStore.addDocuments(chunks, embeddings);
    this.processedSources.add(source);

    return { chunks: chunks.length, source };
  }

  // Query the RAG pipeline with streaming
  async query(
    question: string,
    onChunk?: (chunk: string) => void,
    language: string = 'English'
  ): Promise<{
    answer: string;
    sources: SearchResult[];
  }> {
    if (!this.isReady()) {
      throw new Error('Models not loaded. Please wait for initialization.');
    }

    // Embed the query
    const queryEmbedding = await this.getEmbedding(question);

    // Retrieve relevant chunks
    const results = this.vectorStore.search(queryEmbedding, this.config.topK);

    if (results.length === 0) {
      return {
        answer: 'No relevant documents found in the knowledge base. Please add documents first.',
        sources: [],
      };
    }

    // Build context from retrieved chunks
    const context = results
      .map((r, i) => `[Source ${i + 1}] ${r.chunk.text}`)
      .join('\n\n');

    // Create chat messages for the LLM
    const messages = [
      {
        role: 'system',
        content: `You are a helpful assistant that answers questions based on the provided context. Always ground your answer in the context. If the context doesn't contain enough information, say so clearly.

IMPORTANT: You must respond in ${language}. Translate your answer into ${language} regardless of the language of the context or question.

Context:
${context}`
      },
      {
        role: 'user',
        content: question
      }
    ];

    // Check if engine is still valid before using it
    if (!this.isEngineValid()) {
      throw new Error('LLM engine has been disposed. Please reload the page to reinitialize the models.');
    }

    // Generate response
    let response;
    try {
      response = await this.llmEngine!.chat.completions.create({
        messages: messages as any,
        temperature: 0.7,
        max_tokens: 500,
      });
    } catch (error: any) {
      if (error.message?.includes('disposed') || error.message?.includes('not loaded')) {
        throw new Error('LLM engine has been disposed. Please reload the page to reinitialize the models.');
      }
      throw error;
    }

    const fullResponse = response.choices[0]?.message?.content || 'No response generated.';
    
    // Simulate streaming for better UX
    if (onChunk) {
      const words = fullResponse.split(' ');
      let streamed = '';
      for (const word of words) {
        streamed += word + ' ';
        onChunk(streamed);
        await new Promise(resolve => setTimeout(resolve, 30));
      }
    }

    return { answer: fullResponse, sources: results };
  }

  // Get pipeline status
  getStatus() {
    const info = this.vectorStore.getInfo();
    return {
      isReady: this.isReady(),
      isLoading: this.isLoading,
      collectionName: info.collectionName,
      totalChunks: info.documentCount,
      embeddingDimension: info.embeddingDimension,
      processedSources: Array.from(this.processedSources),
      config: this.config,
    };
  }

  // Clear the knowledge base
  clearKnowledgeBase(): void {
    this.vectorStore.clear();
    this.processedSources.clear();
  }

  // Remove a specific source
  removeSource(source: string): void {
    this.vectorStore.removeBySource(source);
    this.processedSources.delete(source);
  }
}
