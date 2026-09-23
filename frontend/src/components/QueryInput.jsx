import React, { useState } from 'react';
import { Send } from 'lucide-react';

export default function QueryInput({ onSend, disabled }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form className="input-area" onSubmit={handleSubmit}>
      <input
        type="text"
        className="input-field"
        placeholder="Ask a question about candidates, contracts, or SLAs..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      <button
        type="submit"
        className="send-btn"
        disabled={disabled || !text.trim()}
        title="Send Query"
      >
        <Send size={18} />
      </button>
    </form>
  );
}
