import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ChatWindow from './components/ChatWindow';
import PromptChips from './components/PromptChips';
import QueryInput from './components/QueryInput';

const BACKEND_URL = 'http://localhost:8000';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState({
    isOnline: false,
    docCount: 0,
    chunkCount: 0,
  });

  // Check backend health on mount and every 10 seconds
  const checkHealth = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus({
          isOnline: true,
          docCount: data.documents_indexed || 0,
          chunkCount: data.chunks_indexed || 0,
        });
      } else {
        setBackendStatus((prev) => ({ ...prev, isOnline: false }));
      }
    } catch (err) {
      setBackendStatus((prev) => ({ ...prev, isOnline: false }));
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async (userQuery) => {
    setError(null);
    const newMessages = [...messages, { role: 'user', content: userQuery }];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      const response = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userQuery,
          top_k: 3,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned HTTP ${response.status}`);
      }

      const data = await response.json();

      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          content: data.answer,
          citations: data.citations || [],
          retrieved_chunks: data.retrieved_chunks || [],
        },
      ]);
    } catch (err) {
      console.error('Fetch error:', err);
      // Helpful diagnostic for students: distinguish between network/CORS error vs 500 error
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError('Network / CORS error: Unable to reach FastAPI at http://localhost:8000. Ensure the backend is running and allow_origins allows http://localhost:5173.');
      } else {
        setError(err.message || 'An error occurred while communicating with the TrustRAG backend.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Navbar
        isOnline={backendStatus.isOnline}
        docCount={backendStatus.docCount}
        chunkCount={backendStatus.chunkCount}
      />

      <main className="chat-main glass-panel">
        {messages.length === 0 ? (
          <PromptChips onSelectPrompt={handleSendMessage} />
        ) : (
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            error={error}
          />
        )}

        <QueryInput onSend={handleSendMessage} disabled={isLoading} />
      </main>
    </div>
  );
}
