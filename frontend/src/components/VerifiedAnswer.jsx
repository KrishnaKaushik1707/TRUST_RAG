import React, { useState } from 'react';
import { CheckCircle2, AlertTriangle, XCircle, FileText, Info, ChevronDown, ChevronUp } from 'lucide-react';

export default function VerifiedAnswer({ content, verifiedSentences = [] }) {
  const [selectedIdx, setSelectedIdx] = useState(null);

  // If no claim verification data is available (e.g. refusal message), render plain text
  if (!verifiedSentences || verifiedSentences.length === 0) {
    return <div style={{ whiteSpace: 'pre-wrap' }}>{content}</div>;
  }

  // Count stats
  const supportedCount = verifiedSentences.filter((s) => s.label === 'SUPPORTED').length;
  const partialCount = verifiedSentences.filter((s) => s.label === 'PARTIALLY_SUPPORTED').length;
  const unverifiedCount = verifiedSentences.filter((s) => s.label === 'UNVERIFIED').length;

  const handleSentenceClick = (idx) => {
    setSelectedIdx(selectedIdx === idx ? null : idx);
  };

  const getTierConfig = (label) => {
    switch (label) {
      case 'SUPPORTED':
        return {
          icon: <CheckCircle2 size={13} className="text-emerald-400" />,
          color: '#10b981',
          bg: 'rgba(16, 185, 129, 0.14)',
          border: 'rgba(16, 185, 129, 0.35)',
          badgeText: 'Supported',
        };
      case 'PARTIALLY_SUPPORTED':
        return {
          icon: <AlertTriangle size={13} className="text-amber-400" />,
          color: '#f59e0b',
          bg: 'rgba(245, 158, 11, 0.14)',
          border: 'rgba(245, 158, 11, 0.35)',
          badgeText: 'Partial Support',
        };
      case 'UNVERIFIED':
      default:
        return {
          icon: <XCircle size={13} className="text-rose-400" />,
          color: '#ef4444',
          bg: 'rgba(239, 68, 68, 0.14)',
          border: 'rgba(239, 68, 68, 0.35)',
          badgeText: 'Unverified',
        };
    }
  };

  return (
    <div className="verified-answer-container">
      {/* Verification Summary Audit Bar */}
      <div className="verification-audit-bar">
        <div className="audit-bar-title">
          <Info size={14} style={{ color: 'var(--accent-blue)' }} />
          <span>Sentence-Level Claim Verification (NLI)</span>
        </div>
        <div className="audit-pills">
          <span className="audit-pill supported">
            <CheckCircle2 size={12} /> {supportedCount} Supported
          </span>
          {partialCount > 0 && (
            <span className="audit-pill partial">
              <AlertTriangle size={12} /> {partialCount} Partial
            </span>
          )}
          {unverifiedCount > 0 && (
            <span className="audit-pill unverified">
              <XCircle size={12} /> {unverifiedCount} Unverified
            </span>
          )}
        </div>
      </div>

      {/* Sentence-by-sentence highlighted text */}
      <div className="verified-sentences-body">
        {verifiedSentences.map((item, idx) => {
          const cfg = getTierConfig(item.label);
          const isSelected = selectedIdx === idx;

          return (
            <span
              key={idx}
              className={`claim-sentence-pill ${item.label.toLowerCase()} ${isSelected ? 'active-claim' : ''}`}
              onClick={() => handleSentenceClick(idx)}
              title="Click to inspect NLI entailment probabilities & source evidence"
              style={{
                backgroundColor: isSelected ? cfg.bg.replace('0.14', '0.28') : cfg.bg,
                borderColor: cfg.border,
              }}
            >
              <span className="claim-icon" style={{ color: cfg.color }}>
                {cfg.icon}
              </span>
              <span className="claim-text">{item.sentence}</span>
              <span className="claim-confidence-tag" style={{ color: cfg.color }}>
                {(item.entailment_score * 100).toFixed(0)}%
              </span>
              {' '}
            </span>
          );
        })}
      </div>

      {/* Interactive Evidence Inspector Drawer */}
      {selectedIdx !== null && verifiedSentences[selectedIdx] && (
        <div className="claim-inspector-drawer glass-panel">
          {(() => {
            const claim = verifiedSentences[selectedIdx];
            const cfg = getTierConfig(claim.label);

            return (
              <div>
                <div className="inspector-header">
                  <div className="inspector-title" style={{ color: cfg.color }}>
                    {cfg.icon}
                    <span>Claim Audit: {cfg.badgeText}</span>
                  </div>
                  <button
                    className="inspector-close-btn"
                    onClick={() => setSelectedIdx(null)}
                    title="Close Inspector"
                  >
                    ✕
                  </button>
                </div>

                <div className="inspector-claim-quote">
                  "{claim.sentence}"
                </div>

                {/* NLI Probability Breakdown Bar */}
                <div className="nli-scores-grid">
                  <div className="nli-score-card">
                    <div className="nli-score-label">Entailment (Support)</div>
                    <div className="nli-score-val" style={{ color: '#10b981' }}>
                      {(claim.entailment_score * 100).toFixed(1)}%
                    </div>
                    <div className="score-meter-bg">
                      <div
                        className="score-meter-fill"
                        style={{
                          width: `${claim.entailment_score * 100}%`,
                          backgroundColor: '#10b981',
                        }}
                      />
                    </div>
                  </div>

                  <div className="nli-score-card">
                    <div className="nli-score-label">Neutral (No evidence)</div>
                    <div className="nli-score-val" style={{ color: '#94a3b8' }}>
                      {(claim.neutral_score * 100).toFixed(1)}%
                    </div>
                    <div className="score-meter-bg">
                      <div
                        className="score-meter-fill"
                        style={{
                          width: `${claim.neutral_score * 100}%`,
                          backgroundColor: '#64748b',
                        }}
                      />
                    </div>
                  </div>

                  <div className="nli-score-card">
                    <div className="nli-score-label">Contradiction (False)</div>
                    <div className="nli-score-val" style={{ color: '#ef4444' }}>
                      {(claim.contradiction_score * 100).toFixed(1)}%
                    </div>
                    <div className="score-meter-bg">
                      <div
                        className="score-meter-fill"
                        style={{
                          width: `${claim.contradiction_score * 100}%`,
                          backgroundColor: '#ef4444',
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Supporting Source Passage */}
                {claim.supporting_source_file ? (
                  <div className="inspector-evidence-box">
                    <div className="evidence-header">
                      <FileText size={14} style={{ color: 'var(--accent-blue)' }} />
                      <span className="evidence-doc">
                        {claim.supporting_document_name || claim.supporting_source_file}
                      </span>
                      <span className="evidence-page">Page {claim.supporting_page_number || 1}</span>
                    </div>
                    {claim.supporting_text_snippet && (
                      <p className="evidence-snippet">
                        "...{claim.supporting_text_snippet}..."
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="inspector-no-evidence">
                    No retrieved passage logically entails this specific sentence.
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}
