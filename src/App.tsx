import { useState, useRef, useEffect } from 'react';
import { LocalRAGPipeline } from './lib/localRAGPipeline';
import { SearchResult } from './lib/vectorStore';
import { LANGUAGES, LANG_NAME, T, detectLanguage, type LangCode } from './lib/i18n';
import { createVoiceService, checkVoiceSupport } from './lib/voiceService';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources?: SearchResult[];
  isStreaming?: boolean;
}

interface PipelineStatus {
  isReady: boolean;
  isLoading: boolean;
  totalChunks: number;
  processedSources: string[];
  embeddingDimension: number;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [showSources, setShowSources] = useState<string | null>(null);
  const [ingestProgress, setIngestProgress] = useState<string>('');
  const [showIngestPanel, setShowIngestPanel] = useState(false);
  const [pastedText, setPastedText] = useState('');
  const [pastedSourceName, setPastedSourceName] = useState('');
  const [initProgress, setInitProgress] = useState<string>('');
  const [initPercent, setInitPercent] = useState<number>(0);
  const [lang, setLang] = useState<LangCode>(() => {
    const saved = localStorage.getItem('rag-lang');
    return (saved as LangCode) || detectLanguage();
  });
  const tr = T[lang];

  const changeLang = (newLang: LangCode) => {
    setLang(newLang);
    localStorage.setItem('rag-lang', newLang);
  };

  // Voice state
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState({ recognition: false, synthesis: false });
  const voiceServiceRef = useRef<ReturnType<typeof createVoiceService> | null>(null);

  // Initialize voice service
  useEffect(() => {
    const support = checkVoiceSupport();
    setVoiceSupported(support);
    if (support.recognition || support.synthesis) {
      voiceServiceRef.current = createVoiceService();
    }
  }, []);

  const pipelineRef = useRef<LocalRAGPipeline | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize pipeline on mount
  useEffect(() => {
    const initPipeline = async () => {
      pipelineRef.current = new LocalRAGPipeline();
      
      setInitProgress('Initializing models...');
      try {
        await pipelineRef.current.initialize((stage, progress) => {
          setInitProgress(stage);
          setInitPercent(progress);
        });
        
        updateStatus();
        setMessages([{
          id: 'welcome',
          role: 'system',
          content: tr.modelsLoaded
        }]);
      } catch (err: any) {
        setInitProgress(`❌ Error: ${err.message}`);
        setMessages([{
          id: 'error',
          role: 'system',
          content: tr.modelFailed.replace('{error}', err.message)
        }]);
      }
    };

    initPipeline();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Update pipeline status
  const updateStatus = () => {
    if (pipelineRef.current) {
      const s = pipelineRef.current.getStatus();
      setPipelineStatus({
        isReady: s.isReady,
        isLoading: s.isLoading,
        totalChunks: s.totalChunks,
        processedSources: s.processedSources,
        embeddingDimension: s.embeddingDimension,
      });
    }
  };

  // Handle PDF upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || !pipelineRef.current) return;

    setIsLoading(true);
    for (const file of Array.from(files)) {
      try {
        setIngestProgress(`Processing ${file.name}...`);
        const result = await pipelineRef.current.ingestPDF(file, (stage) => {
          setIngestProgress(`${file.name}: ${stage}`);
        });
        setIngestProgress(`✓ ${file.name}: ${result.chunks} chunks indexed`);
        setMessages((prev) => [
          ...prev,
          {
            id: `sys-${Date.now()}`,
            role: 'system',
            content: tr.indexedMsg.replace('{source}', file.name).replace('{chunks}', String(result.chunks)),
          },
        ]);
      } catch (err: any) {
        setIngestProgress(`✗ Error: ${err.message}`);
        setMessages((prev) => [
          ...prev,
          {
            id: `sys-err-${Date.now()}`,
            role: 'system',
            content: `❌ Failed to process "${file.name}": ${err.message}`,
          },
        ]);
      }
    }
    updateStatus();
    setIsLoading(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
    setTimeout(() => setIngestProgress(''), 4000);
  };

  // Handle text paste
  const handleTextIngest = async () => {
    if (!pastedText.trim() || !pipelineRef.current) return;
    setIsLoading(true);
    try {
      const source = pastedSourceName.trim() || `text-${Date.now()}`;
      setIngestProgress(`Indexing "${source}"...`);
      const result = await pipelineRef.current.ingestText(pastedText, source, (stage) => {
        setIngestProgress(`${source}: ${stage}`);
      });
      setIngestProgress(`✓ "${source}": ${result.chunks} chunks indexed`);
      setMessages((prev) => [
        ...prev,
        {
          id: `sys-${Date.now()}`,
          role: 'system',
          content: tr.indexedMsg.replace('{source}', source).replace('{chunks}', String(result.chunks)),
        },
      ]);
      setPastedText('');
      setPastedSourceName('');
      setShowIngestPanel(false);
      updateStatus();
    } catch (err: any) {
      setIngestProgress(`✗ Error: ${err.message}`);
    }
    setIsLoading(false);
    setTimeout(() => setIngestProgress(''), 4000);
  };

  // Send message and get RAG response
  const handleSend = () => {
    if (!input.trim()) return;
    const text = input;
    setInput('');
    handleSendWithText(text);
  };

  // Clear knowledge base
  const clearKnowledge = () => {
    pipelineRef.current?.clearKnowledgeBase();
    updateStatus();
    setMessages((prev) => [
      ...prev,
      {
        id: `sys-${Date.now()}`,
        role: 'system',
        content: tr.clearedMsg,
      },
    ]);
  };

  const removeSource = (source: string) => {
    pipelineRef.current?.removeSource(source);
    updateStatus();
  };

  // Voice handlers
  const startVoiceInput = () => {
    if (!voiceServiceRef.current || !voiceSupported.recognition) return;
    
    setIsListening(true);
    voiceServiceRef.current.startListening(
      lang,
      (text) => {
        setInput(text);
        setIsListening(false);
        // Auto-send in voice mode
        if (voiceMode && text.trim()) {
          setTimeout(() => {
            setInput('');
            handleSendWithText(text);
          }, 500);
        }
      },
      (error) => {
        console.error('Voice input error:', error);
        setIsListening(false);
        setMessages((prev) => [
          ...prev,
          {
            id: `sys-voice-${Date.now()}`,
            role: 'system',
            content: `${tr.voiceError}: ${error}`,
          },
        ]);
      }
    );
  };

  const stopVoiceInput = () => {
    if (!voiceServiceRef.current) return;
    voiceServiceRef.current.stopListening();
    setIsListening(false);
  };

  const speakMessage = (text: string) => {
    if (!voiceServiceRef.current || !voiceSupported.synthesis) return;
    
    setIsSpeaking(true);
    voiceServiceRef.current.speak(text, lang, () => {
      setIsSpeaking(false);
    });
  };

  const stopSpeaking = () => {
    if (!voiceServiceRef.current) return;
    voiceServiceRef.current.stopSpeaking();
    setIsSpeaking(false);
  };

  const toggleVoiceMode = () => {
    setVoiceMode(!voiceMode);
  };

  // Handle send with specific text (for voice input)
  const handleSendWithText = async (text: string) => {
    if (!text.trim() || isLoading || !pipelineRef.current) return;

    if (!pipelineRef.current.isReady()) {
      setMessages((prev) => [
        ...prev,
        {
          id: `sys-${Date.now()}`,
          role: 'system',
          content: '⚠️ Models are still loading. Please wait...',
        },
      ]);
      return;
    }

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text.trim(),
    };

    const assistantMessage: Message = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsLoading(true);

    const question = userMessage.content;

    try {
      let fullAnswer = '';

      const result = await pipelineRef.current.query(
        question,
        (chunk) => {
          fullAnswer = chunk;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id ? { ...m, content: fullAnswer } : m
            )
          );
        },
        LANG_NAME[lang]
      );

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessage.id
            ? { ...m, content: result.answer, sources: result.sources, isStreaming: false }
            : m
        )
      );

      // Auto-speak in voice mode
      if (voiceMode && voiceSupported.synthesis) {
        setTimeout(() => speakMessage(result.answer), 300);
      }
    } catch (err: any) {
      const isDisposed = err.message?.includes('disposed') || err.message?.includes('not loaded');
      
      if (isDisposed && pipelineRef.current) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessage.id
              ? { ...m, content: tr.reloadMsg, isStreaming: false }
              : m
          )
        );
        
        try {
          await pipelineRef.current.reloadEngine((stage, progress) => {
            setInitProgress(stage);
            setInitPercent(progress);
          });
          
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, content: tr.retryMsg, isStreaming: true }
                : m
            )
          );
          
          const retryResult = await pipelineRef.current.query(question, (chunk) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessage.id ? { ...m, content: chunk } : m
              )
            );
          }, LANG_NAME[lang]);
          
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, content: retryResult.answer, sources: retryResult.sources, isStreaming: false }
                : m
            )
          );

          // Auto-speak in voice mode
          if (voiceMode && voiceSupported.synthesis) {
            setTimeout(() => speakMessage(retryResult.answer), 300);
          }
        } catch (reloadErr: any) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, content: tr.reloadFailed.replace('{error}', reloadErr.message), isStreaming: false }
                : m
            )
          );
        }
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessage.id
              ? { ...m, content: `❌ Error: ${err.message}`, isStreaming: false }
              : m
          )
        );
      }
    }

    setIsLoading(false);
  };

  const isReady = pipelineStatus?.isReady ?? false;
  const isLoadingModels = pipelineStatus?.isLoading ?? true;

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/90 backdrop-blur-sm px-4 py-3 flex items-center justify-between flex-shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-purple-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-purple-900/30">
            <span className="text-lg">🦙</span>
          </div>
          <div>
            <h1 className="text-base font-bold text-white leading-tight">{tr.appTitle}</h1>
            <p className="text-[11px] text-gray-500">{tr.appSubtitle}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Status */}
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
            isReady ? 'bg-green-900/40 text-green-400 border border-green-800/60' :
            isLoadingModels ? 'bg-yellow-900/40 text-yellow-400 border border-yellow-800/60' :
            'bg-red-900/40 text-red-400 border border-red-800/60'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              isReady ? 'bg-green-400 animate-pulse' :
              isLoadingModels ? 'bg-yellow-400 animate-pulse' :
              'bg-red-400'
            }`}></span>
            <span className="hidden sm:inline">
              {isReady ? tr.statusReady : isLoadingModels ? tr.statusLoading : tr.statusError}
            </span>
          </div>

          {/* KB Info */}
          {pipelineStatus && pipelineStatus.totalChunks > 0 && (
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs bg-purple-900/40 text-purple-300 border border-purple-800/60">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
              {pipelineStatus.totalChunks} {tr.chunks}
            </div>
          )}

          {/* Language Selector */}
          <div className="relative">
            <select
              value={lang}
              onChange={(e) => changeLang(e.target.value as LangCode)}
              className="appearance-none bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-3 py-1.5 pr-7 focus:outline-none focus:border-purple-500 cursor-pointer"
              style={{ direction: 'ltr' }}
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.flag} {l.native}
                </option>
              ))}
            </select>
            <svg className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-gray-400 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </header>

      {/* Loading Banner */}
      {isLoadingModels && !isReady && (
        <div className="bg-blue-950/40 border-b border-blue-900/50 px-4 py-3 flex-shrink-0">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-blue-400 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <div className="flex-1">
                <p className="text-sm text-blue-200 font-medium">{initProgress}</p>
                <div className="mt-1 w-full bg-gray-800 rounded-full h-1.5">
                  <div className="bg-blue-500 h-1.5 rounded-full transition-all duration-300" style={{ width: `${initPercent}%` }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-72 border-r border-gray-800 bg-gray-900/50 flex-col flex-shrink-0 hidden md:flex overflow-hidden">
          <div className="p-4 border-b border-gray-800 flex-shrink-0">
            <h2 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              {tr.knowledgeBase}
            </h2>

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading || !isReady}
              className="w-full mb-2 px-3 py-2.5 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:text-gray-500 text-white text-xs font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              {tr.uploadPDF}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              multiple
              onChange={handleFileUpload}
              className="hidden"
            />

            <button
              onClick={() => setShowIngestPanel(!showIngestPanel)}
              disabled={isLoading || !isReady}
              className="w-full px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white text-xs rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              {tr.pasteText}
            </button>
          </div>

          {/* Paste Text Panel */}
          {showIngestPanel && (
            <div className="p-4 border-b border-gray-800 flex-shrink-0">
              <input
                type="text"
                value={pastedSourceName}
                onChange={(e) => setPastedSourceName(e.target.value)}
                placeholder={tr.sourceName}
                className="w-full mb-2 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500"
              />
              <textarea
                value={pastedText}
                onChange={(e) => setPastedText(e.target.value)}
                placeholder={tr.pastePlaceholder}
                rows={5}
                className="w-full mb-2 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500 resize-none"
              />
              <button
                onClick={handleTextIngest}
                disabled={isLoading || !pastedText.trim() || !isReady}
                className="w-full px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:text-gray-500 text-white text-xs rounded-lg transition-colors"
              >
                {tr.indexText}
              </button>
            </div>
          )}

          {/* Progress */}
          {ingestProgress && (
            <div className="px-4 py-2 border-b border-gray-800 bg-gray-800/50 flex-shrink-0">
              <p className="text-xs text-gray-300 truncate">{ingestProgress}</p>
            </div>
          )}

          {/* Indexed Sources */}
          <div className="flex-1 overflow-y-auto p-4">
            {pipelineStatus && pipelineStatus.processedSources.length > 0 ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400 font-medium">{tr.indexed} ({pipelineStatus.processedSources.length})</span>
                  <button
                    onClick={clearKnowledge}
                    className="text-xs text-red-400 hover:text-red-300 transition-colors"
                  >
                    {tr.clearAll}
                  </button>
                </div>
                {pipelineStatus.processedSources.map((source) => (
                  <div key={source} className="flex items-center gap-2 bg-gray-800/80 rounded-lg px-3 py-2 group">
                    <svg className="w-4 h-4 text-green-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-xs text-gray-300 truncate flex-1">{source}</span>
                    <button
                      onClick={() => removeSource(source)}
                      className="text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                      title="Remove"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <div className="w-16 h-16 bg-gray-800 rounded-2xl flex items-center justify-center mx-auto mb-3">
                  <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <p className="text-xs text-gray-500 font-medium">{tr.noDocuments}</p>
                <p className="text-[10px] text-gray-600 mt-1">{tr.noDocumentsHint}</p>
              </div>
            )}
          </div>
        </aside>

        {/* Chat Area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-6">
            <div className="max-w-3xl mx-auto space-y-4">
              {messages.length === 0 && (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">🦙</div>
                  <h2 className="text-2xl font-bold text-white mb-2">{tr.welcomeTitle}</h2>
                  <p className="text-gray-400 mb-8 max-w-md mx-auto text-sm">
                    {tr.welcomeDesc}
                  </p>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 max-w-lg mx-auto mb-8">
                    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                      <div className="text-2xl mb-2">📄</div>
                      <p className="text-xs text-gray-400">{tr.step1}</p>
                    </div>
                    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                      <div className="text-2xl mb-2">🔍</div>
                      <p className="text-xs text-gray-400">{tr.step2}</p>
                    </div>
                    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                      <div className="text-2xl mb-2">💬</div>
                      <p className="text-xs text-gray-400">{tr.step3}</p>
                    </div>
                  </div>

                  {/* Mobile upload */}
                  <div className="md:hidden space-y-2">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isLoading || !isReady}
                      className="px-5 py-2.5 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 text-white text-sm rounded-lg transition-colors"
                    >
                      {tr.uploadPDF}
                    </button>
                  </div>
                </div>
              )}

              {messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] ${
                    msg.role === 'user'
                      ? 'bg-purple-600 text-white rounded-2xl rounded-br-sm px-4 py-3'
                      : msg.role === 'system'
                      ? 'bg-gray-800/60 border border-gray-700/50 text-gray-300 rounded-xl px-4 py-3 text-sm'
                      : 'bg-gray-800 text-gray-100 rounded-2xl rounded-bl-sm px-4 py-3'
                  }`}>
                    <div className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</div>
                    
                    {/* Sources */}
                    {msg.sources && msg.sources.length > 0 && msg.role === 'assistant' && !msg.isStreaming && (
                      <div className="mt-3 pt-3 border-t border-gray-700/50">
                        <button
                          onClick={() => setShowSources(showSources === msg.id ? null : msg.id)}
                          className="flex items-center gap-1.5 text-xs text-purple-400 hover:text-purple-300 transition-colors"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          {msg.sources.length} {tr.sourcesRetrieved} {showSources === msg.id ? '▲' : '▼'}
                        </button>
                        {showSources === msg.id && (
                          <div className="mt-2 space-y-2">
                            {msg.sources.map((source, idx) => (
                              <div key={idx} className="bg-gray-900/80 rounded-lg p-3 border border-gray-700/50">
                                <div className="flex items-center justify-between mb-1.5">
                                  <span className="text-[10px] text-purple-400 font-medium">
                                    {tr.source} {idx + 1}: {source.chunk.metadata.source}
                                    {source.chunk.metadata.page && ` (p.${source.chunk.metadata.page})`}
                                  </span>
                                  <span className="text-[10px] bg-purple-900/40 text-purple-300 px-1.5 py-0.5 rounded">
                                    {(source.score * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <p className="text-xs text-gray-400 leading-relaxed line-clamp-3">{source.chunk.text}</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {msg.isStreaming && (
                      <span className="inline-block w-1.5 h-4 bg-purple-400 animate-pulse ml-0.5 align-middle rounded-sm"></span>
                    )}

                    {/* Speak Button for Assistant Messages */}
                    {msg.role === 'assistant' && !msg.isStreaming && msg.content && voiceSupported.synthesis && (
                      <div className="mt-2 flex justify-end">
                        <button
                          onClick={() => isSpeaking ? stopSpeaking() : speakMessage(msg.content)}
                          className={`text-xs px-2 py-1 rounded transition-colors flex items-center gap-1 ${
                            isSpeaking
                              ? 'bg-red-600 hover:bg-red-700 text-white'
                              : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                          }`}
                          title={isSpeaking ? tr.stopSpeaking : tr.speak}
                        >
                          {isSpeaking ? (
                            <>
                              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M6 6h12v12H6z"/>
                              </svg>
                              {tr.stopSpeaking}
                            </>
                          ) : (
                            <>
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                              </svg>
                              {tr.speak}
                            </>
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-800 bg-gray-900/50 px-4 py-4 flex-shrink-0">
            <div className="max-w-3xl mx-auto">
              <div className="flex gap-2">
                {/* Voice Input Button */}
                {voiceSupported.recognition && (
                  <button
                    onClick={isListening ? stopVoiceInput : startVoiceInput}
                    disabled={isLoading || !isReady}
                    className={`px-4 py-3 rounded-xl transition-colors flex items-center justify-center ${
                      isListening
                        ? 'bg-red-600 hover:bg-red-700 animate-pulse'
                        : 'bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600'
                    } text-white`}
                    title={isListening ? tr.voiceListening : tr.voiceInput}
                  >
                    {isListening ? (
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                      </svg>
                    ) : (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                      </svg>
                    )}
                  </button>
                )}
                
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder={
                    isListening
                      ? tr.voiceListening
                      : !isReady
                      ? tr.placeholderWaiting
                      : pipelineStatus && pipelineStatus.totalChunks === 0
                      ? tr.placeholderEmpty
                      : tr.placeholderAsk
                  }
                  disabled={isLoading || !isReady || isListening}
                  className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30 disabled:opacity-50 transition-all"
                />
                <button
                  onClick={handleSend}
                  disabled={isLoading || !input.trim() || !isReady}
                  className="px-4 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-xl transition-colors flex items-center justify-center"
                >
                  {isLoading ? (
                    <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  )}
                </button>
              </div>
              <div className="flex items-center justify-between mt-2">
                <p className="text-[10px] text-gray-600">
                  Llama 3.2 1B • all-MiniLM-L6-v2 • {tr.footerLocal}
                </p>
                {/* Voice Mode Toggle */}
                {(voiceSupported.recognition || voiceSupported.synthesis) && (
                  <button
                    onClick={toggleVoiceMode}
                    className={`text-[10px] px-2 py-1 rounded transition-colors ${
                      voiceMode
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                    title={tr.voiceAssistant}
                  >
                    🎤 {tr.voiceAssistant}
                  </button>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
