// Voice interaction service using Web Speech API
// Supports speech recognition (STT) and text-to-speech (TTS) in multiple languages

import { type LangCode } from './i18n';

// Language code mapping for Web Speech API
const SPEECH_LANG_MAP: Record<LangCode, string> = {
  en: 'en-US',
  es: 'es-ES',
  zh: 'zh-CN',
  hi: 'hi-IN',
  ar: 'ar-SA',
  fr: 'fr-FR',
  de: 'de-DE',
  ja: 'ja-JP',
  pt: 'pt-BR',
  ru: 'ru-RU',
};

// TTS voice mapping
const TTS_LANG_MAP: Record<LangCode, string> = {
  en: 'en-US',
  es: 'es-ES',
  zh: 'zh-CN',
  hi: 'hi-IN',
  ar: 'ar-SA',
  fr: 'fr-FR',
  de: 'de-DE',
  ja: 'ja-JP',
  pt: 'pt-BR',
  ru: 'ru-RU',
};

export interface VoiceService {
  isSupported: boolean;
  isListening: boolean;
  isSpeaking: boolean;
  startListening: (lang: LangCode, onResult: (text: string) => void, onError?: (error: string) => void) => void;
  stopListening: () => void;
  speak: (text: string, lang: LangCode, onEnd?: () => void) => void;
  stopSpeaking: () => void;
}

export function createVoiceService(): VoiceService {
  let recognition: any = null;
  let isListening = false;
  let isSpeaking = false;

  // Check if speech recognition is supported
  const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  const isSupported = !!SpeechRecognition && 'speechSynthesis' in window;

  // Initialize recognition
  if (isSupported) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
  }

  return {
    isSupported,
    get isListening() { return isListening; },
    get isSpeaking() { return isSpeaking; },

    startListening: (lang: LangCode, onResult: (text: string) => void, onError?: (error: string) => void) => {
      if (!recognition) {
        onError?.('Speech recognition not supported');
        return;
      }

      recognition.lang = SPEECH_LANG_MAP[lang];
      
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        onResult(transcript);
        isListening = false;
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        onError?.(`Speech recognition error: ${event.error}`);
        isListening = false;
      };

      recognition.onend = () => {
        isListening = false;
      };

      try {
        recognition.start();
        isListening = true;
      } catch (err: any) {
        onError?.(`Failed to start recognition: ${err.message}`);
      }
    },

    stopListening: () => {
      if (recognition && isListening) {
        recognition.stop();
        isListening = false;
      }
    },

    speak: (text: string, lang: LangCode, onEnd?: () => void) => {
      if (!('speechSynthesis' in window)) {
        console.warn('Text-to-speech not supported');
        return;
      }

      // Cancel any ongoing speech
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = TTS_LANG_MAP[lang];
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;

      // Try to find a voice matching the language
      const voices = window.speechSynthesis.getVoices();
      const targetLang = TTS_LANG_MAP[lang];
      const matchingVoice = voices.find(v => v.lang.startsWith(targetLang.split('-')[0]));
      
      if (matchingVoice) {
        utterance.voice = matchingVoice;
      }

      utterance.onstart = () => {
        isSpeaking = true;
      };

      utterance.onend = () => {
        isSpeaking = false;
        onEnd?.();
      };

      utterance.onerror = (event: any) => {
        console.error('Speech synthesis error:', event.error);
        isSpeaking = false;
      };

      window.speechSynthesis.speak(utterance);
    },

    stopSpeaking: () => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        isSpeaking = false;
      }
    },
  };
}

// Check if voice features are available
export function checkVoiceSupport(): { recognition: boolean; synthesis: boolean } {
  const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  return {
    recognition: !!SpeechRecognition,
    synthesis: 'speechSynthesis' in window,
  };
}
