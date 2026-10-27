// Real PDF text extraction using pdfjs-dist
// This is the same approach used by pypdf in the Python version

import * as pdfjsLib from 'pdfjs-dist';

// Configure the worker - use CDN for production builds
pdfjsLib.GlobalWorkerOptions.workerSrc = 
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

export interface PDFPage {
  pageNumber: number;
  text: string;
}

export async function extractTextFromPDF(
  file: File,
  onProgress?: (current: number, total: number) => void
): Promise<PDFPage[]> {
  const arrayBuffer = await file.arrayBuffer();
  const loadingTask = pdfjsLib.getDocument({
    data: new Uint8Array(arrayBuffer),
    useWorkerFetch: false,
    isEvalSupported: false,
    useSystemFonts: true,
  });

  const pdf = await loadingTask.promise;
  const pages: PDFPage[] = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();
    const text = textContent.items
      .map((item: any) => ('str' in item ? item.str : ''))
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (text) {
      pages.push({ pageNumber: i, text });
    }

    if (onProgress) onProgress(i, pdf.numPages);
  }

  return pages;
}

// Also support extracting text from a URL (like the Python version loads ThaiRecipes.pdf)
export async function extractTextFromPDFUrl(
  url: string,
  onProgress?: (current: number, total: number) => void
): Promise<PDFPage[]> {
  const loadingTask = pdfjsLib.getDocument({
    url,
    useWorkerFetch: false,
    isEvalSupported: false,
    useSystemFonts: true,
  });

  const pdf = await loadingTask.promise;
  const pages: PDFPage[] = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();
    const text = textContent.items
      .map((item: any) => ('str' in item ? item.str : ''))
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (text) {
      pages.push({ pageNumber: i, text });
    }

    if (onProgress) onProgress(i, pdf.numPages);
  }

  return pages;
}
