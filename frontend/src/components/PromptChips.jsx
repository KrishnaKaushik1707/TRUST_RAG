import React from 'react';
import { ArrowRight, Sparkles } from 'lucide-react';

const SUGGESTED_PROMPTS = [
  "What are the skills of Kaushik in Kaushik_ML_DEV_Resume.pdf?",
  "What are the technical skills and achievements of Meghana?",
  "What is the experience and hackathon background of Shyam?",
  "What are the service credits and penalty for uptime falling below 95%?",
];

export default function PromptChips({ onSelectPrompt }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Sparkles size={28} />
      </div>
      <h3>Ask Anything About Your Documents</h3>
      <p>
        TrustRAG uses hybrid dense-sparse retrieval to pinpoint exact answers and cite every single claim back to its source page.
      </p>

      <div className="chips-grid">
        {SUGGESTED_PROMPTS.map((prompt, i) => (
          <button
            key={i}
            className="chip-button"
            onClick={() => onSelectPrompt(prompt)}
          >
            <span>{prompt}</span>
            <ArrowRight size={14} style={{ color: 'var(--accent-blue)', opacity: 0.8 }} />
          </button>
        ))}
      </div>
    </div>
  );
}
