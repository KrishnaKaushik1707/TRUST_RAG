import React from 'react';
import { ShieldCheck, Cpu } from 'lucide-react';

export default function Navbar({ isOnline, docCount, chunkCount }) {
  return (
    <header className="app-header glass-panel">
      <div className="brand">
        <div className="brand-icon">
          <ShieldCheck size={22} />
        </div>
        <div>
          <h1 className="brand-title">TrustRAG</h1>
          <p className="brand-subtitle">Document Intelligence with Grounded Citations</p>
        </div>
      </div>

      <div className="header-badges">
        <div className="status-badge">
          <span className={`status-dot ${isOnline ? 'online' : 'offline'}`} />
          <span>{isOnline ? 'Backend Online' : 'Backend Disconnected'}</span>
        </div>

        {isOnline && (
          <div className="status-badge" title="Indexed Documents">
            <Cpu size={14} style={{ color: 'var(--accent-blue)' }} />
            <span>{docCount} Docs / {chunkCount} Chunks</span>
          </div>
        )}
      </div>
    </header>
  );
}
