import React, { useState } from 'react';
import { Database, ChevronDown, ChevronUp } from 'lucide-react';

export default function DiagnosticDrawer({ chunks }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!chunks || chunks.length === 0) return null;

  return (
    <div style={{ marginTop: '12px', borderTop: '1px dashed rgba(148, 163, 184, 0.2)', paddingTop: '10px' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          background: 'none',
          border: 'none',
          color: 'var(--text-muted)',
          fontSize: '0.74rem',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        <Database size={12} />
        <span>{isOpen ? 'Hide' : 'Inspect'} Retrieval Diagnostics ({chunks.length} candidate passages)</span>
        {isOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {isOpen && (
        <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {chunks.map((ch, idx) => (
            <div
              key={idx}
              style={{
                background: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid rgba(148, 163, 184, 0.1)',
                borderRadius: '8px',
                padding: '8px 12px',
                fontSize: '0.75rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', color: 'var(--accent-blue)' }}>
                <span><b>Rank #{ch.final_rank}:</b> {ch.source_file} (Page {ch.page_number})</span>
                <span>Re-Rank: {ch.rerank_score} | Dense #{ch.dense_rank || 'N/A'} | BM25 #{ch.sparse_rank || 'N/A'}</span>
              </div>
              <div style={{ color: 'var(--text-secondary)', fontFamily: 'monospace', fontSize: '0.72rem' }}>
                {ch.text.slice(0, 140)}...
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
