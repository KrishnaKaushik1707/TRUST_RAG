import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp } from 'lucide-react';

export default function CitationCard({ citation, index }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div style={{ display: 'inline-block', margin: '4px' }}>
      <div 
        className="citation-card" 
        onClick={() => setIsOpen(!isOpen)}
        title="Click to view supporting passage"
      >
        <FileText size={13} style={{ color: 'var(--accent-blue)' }} />
        <span className="citation-file">{citation.source_file}</span>
        <span className="citation-page">Page {citation.page_number}</span>
        {isOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </div>

      {isOpen && (
        <div 
          style={{
            background: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: '8px',
            padding: '10px 14px',
            marginTop: '6px',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            maxWidth: '520px',
            boxShadow: '0 6px 20px rgba(0, 0, 0, 0.5)',
            animation: 'fadeIn 0.2s ease-out',
          }}
        >
          <div style={{ color: 'var(--accent-blue)', fontWeight: 600, marginBottom: '4px' }}>
            Excerpt from {citation.document_name} (Page {citation.page_number}):
          </div>
          <div style={{ fontStyle: 'italic', lineHeight: 1.5 }}>
            "{citation.text_snippet}..."
          </div>
        </div>
      )}
    </div>
  );
}
