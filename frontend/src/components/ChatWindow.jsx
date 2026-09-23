import React, { useRef, useEffect } from 'react';
import { User, Bot, Loader2, AlertCircle } from 'lucide-react';
import CitationCard from './CitationCard';
import DiagnosticDrawer from './DiagnosticDrawer';
import VerifiedAnswer from './VerifiedAnswer';

export default function ChatWindow({ messages, isLoading, error }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="messages-area">
      {messages.map((msg, idx) => (
        <div key={idx} className={`message-row ${msg.role}`}>
          <div className={`avatar ${msg.role}`}>
            {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
          </div>

          <div className="bubble">
            {msg.role === 'assistant' ? (
              <VerifiedAnswer
                content={msg.content}
                verifiedSentences={msg.verified_sentences || []}
              />
            ) : (
              <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
            )}

            {/* Citations section if assistant message has citations */}
            {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
              <div className="citations-container">
                <div className="citations-header">
                  <span>Source Evidence Citations</span>
                </div>
                <div className="citation-badges-list">
                  {msg.citations.map((cite, cIdx) => (
                    <CitationCard key={cIdx} citation={cite} index={cIdx} />
                  ))}
                </div>
                {msg.retrieved_chunks && <DiagnosticDrawer chunks={msg.retrieved_chunks} />}
              </div>
            )}
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="message-row assistant">
          <div className="avatar assistant">
            <Bot size={18} />
          </div>
          <div className="bubble" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)' }}>
            <Loader2 size={16} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
            <span>Searching hybrid index & synthesizing answer...</span>
          </div>
        </div>
      )}

      {error && (
        <div className="error-banner">
          <AlertCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <b>Connection / Query Error:</b>
            <div style={{ marginTop: '2px' }}>{error}</div>
            <div style={{ marginTop: '6px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Check if the backend is running at <code>http://localhost:8000</code> and CORS allow_origins contains your frontend port.
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
