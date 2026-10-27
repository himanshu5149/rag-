// Real in-memory vector store with cosine similarity search
// This is the equivalent of Qdrant in the Python version
// Stores embeddings and performs real vector similarity search

import { TextChunk } from './chunker';

export interface VectorDocument {
  id: string;
  chunk: TextChunk;
  embedding: number[];
}

export interface SearchResult {
  chunk: TextChunk;
  score: number;
  embedding: number[];
}

export class VectorStore {
  private documents: VectorDocument[] = [];
  private collectionName: string;

  constructor(collectionName: string = 'rag-collection') {
    this.collectionName = collectionName;
  }

  // Add documents with their embeddings to the store
  addDocuments(chunks: TextChunk[], embeddings: number[][]): void {
    if (chunks.length !== embeddings.length) {
      throw new Error('Chunks and embeddings must have the same length');
    }

    for (let i = 0; i < chunks.length; i++) {
      this.documents.push({
        id: chunks[i].id,
        chunk: chunks[i],
        embedding: embeddings[i],
      });
    }
  }

  // Search for the most similar documents using cosine similarity
  search(queryEmbedding: number[], topK: number = 4): SearchResult[] {
    if (this.documents.length === 0) {
      return [];
    }

    // Calculate cosine similarity for all documents
    const results: SearchResult[] = this.documents.map((doc) => ({
      chunk: doc.chunk,
      score: cosineSimilarity(queryEmbedding, doc.embedding),
      embedding: doc.embedding,
    }));

    // Sort by score descending and return top K
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  // Get collection info
  getInfo() {
    return {
      collectionName: this.collectionName,
      documentCount: this.documents.length,
      embeddingDimension: this.documents.length > 0 ? this.documents[0].embedding.length : 0,
    };
  }

  // Clear all documents
  clear(): void {
    this.documents = [];
  }

  // Remove documents from a specific source
  removeBySource(source: string): void {
    this.documents = this.documents.filter(
      (doc) => doc.chunk.metadata.source !== source
    );
  }

  // Get all documents
  getAllDocuments(): VectorDocument[] {
    return [...this.documents];
  }
}

// Cosine similarity between two vectors
// This is the same metric used by Qdrant by default
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimension');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) return 0;

  return dotProduct / magnitude;
}

// Export the similarity function for testing
export { cosineSimilarity };
