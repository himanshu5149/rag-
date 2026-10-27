// Real text chunking implementation
// Splits documents into overlapping chunks for RAG retrieval
// Mirrors the chunking behavior of Agno's Knowledge class

export interface TextChunk {
  id: string;
  text: string;
  metadata: {
    source: string;
    page?: number;
    chunkIndex: number;
    startChar: number;
    endChar: number;
  };
}

export interface ChunkOptions {
  chunkSize: number;    // Number of characters per chunk
  chunkOverlap: number; // Number of overlapping characters between chunks
  source: string;       // Source document name
}

const DEFAULT_OPTIONS: ChunkOptions = {
  chunkSize: 1000,
  chunkOverlap: 200,
  source: 'document',
};

// Split text into chunks with overlap
export function chunkText(
  text: string,
  options: Partial<ChunkOptions> = {}
): TextChunk[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const chunks: TextChunk[] = [];

  if (!text.trim()) return chunks;

  // Split by paragraphs first for better semantic boundaries
  const paragraphs = text.split(/\n\s*\n/);
  let currentChunk = '';
  let currentStart = 0;
  let chunkIndex = 0;
  let charPosition = 0;

  for (const paragraph of paragraphs) {
    const trimmedParagraph = paragraph.trim();
    if (!trimmedParagraph) continue;

    // If adding this paragraph exceeds chunk size, save current chunk
    if (currentChunk.length + trimmedParagraph.length > opts.chunkSize && currentChunk.length > 0) {
      chunks.push({
        id: `${opts.source}-chunk-${chunkIndex}`,
        text: currentChunk.trim(),
        metadata: {
          source: opts.source,
          chunkIndex,
          startChar: currentStart,
          endChar: currentStart + currentChunk.length,
        },
      });
      chunkIndex++;

      // Keep overlap from end of previous chunk
      const overlapText = currentChunk.slice(-opts.chunkOverlap);
      currentStart = charPosition - overlapText.length;
      currentChunk = overlapText + '\n\n' + trimmedParagraph;
    } else {
      if (currentChunk) {
        currentChunk += '\n\n' + trimmedParagraph;
      } else {
        currentChunk = trimmedParagraph;
        currentStart = charPosition;
      }
    }

    charPosition += trimmedParagraph.length + 2; // +2 for \n\n
  }

  // Don't forget the last chunk
  if (currentChunk.trim()) {
    chunks.push({
      id: `${opts.source}-chunk-${chunkIndex}`,
      text: currentChunk.trim(),
      metadata: {
        source: opts.source,
        chunkIndex,
        startChar: currentStart,
        endChar: currentStart + currentChunk.length,
      },
    });
  }

  return chunks;
}

// Chunk pages from a PDF, preserving page numbers in metadata
export function chunkPages(
  pages: { pageNumber: number; text: string }[],
  options: Partial<ChunkOptions> = {}
): TextChunk[] {
  const allChunks: TextChunk[] = [];

  for (const page of pages) {
    const pageChunks = chunkText(page.text, options);
    for (const chunk of pageChunks) {
      chunk.metadata.page = page.pageNumber;
      chunk.metadata.source = options.source || 'document';
    }
    allChunks.push(...pageChunks);
  }

  // Re-index all chunks
  allChunks.forEach((chunk, idx) => {
    chunk.id = `${chunk.metadata.source}-chunk-${idx}`;
    chunk.metadata.chunkIndex = idx;
  });

  return allChunks;
}
