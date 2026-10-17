import React from 'react';
import { Cpu, Database, FlaskConical, GitBranch } from 'lucide-react';
import type { HardwareInfo, HealthInfo } from '../types';

interface HeaderProps {
  health: HealthInfo | null;
  hardware: HardwareInfo | null;
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  hardware,
  activeTab,
  setActiveTab,
}) => {
  return (
    <header style={{
      backgroundColor: '#FFFFFF',
      borderBottom: '1px solid var(--border-soft)',
      padding: '16px 32px',
      position: 'sticky',
      top: 0,
      zIndex: 50,
    }}>
      <div style={{
        maxWidth: 1300,
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 16,
      }}>
        {/* Brand identity */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 42,
            height: 42,
            backgroundColor: 'var(--sage-light)',
            border: '1px solid var(--border-soft)',
            borderRadius: 8,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--sage-base)',
          }}>
            <FlaskConical size={22} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h1 style={{
                fontSize: 20,
                fontWeight: 700,
                color: 'var(--text-charcoal)',
                letterSpacing: '-0.02em',
                fontFamily: 'serif',
              }}>
                ChemRxn-Intelligence
              </h1>
              <span style={{
                fontSize: 11,
                fontWeight: 600,
                padding: '2px 8px',
                borderRadius: 12,
                backgroundColor: 'var(--sage-light)',
                color: 'var(--sage-dark)',
                border: '1px solid var(--border-soft)',
              }}>
                v0.1.0 Research Core
              </span>
            </div>
            <p style={{
              fontSize: 12,
              color: 'var(--text-secondary)',
              marginTop: 1,
            }}>
              Organic Reaction Prediction & Synthesis Platform • Scratch Neural Architectures
            </p>
          </div>
        </div>

        {/* Telemetry pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '6px 14px',
          backgroundColor: 'var(--bg-subtle)',
          borderRadius: 8,
          border: '1px solid var(--border-soft)',
          fontSize: 12,
          color: 'var(--text-secondary)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: health?.status === 'ok' ? '#4E8752' : '#8A9A86',
            }} />
            <span style={{ fontWeight: 600, color: 'var(--text-charcoal)' }}>
              {hardware?.cuda_available ? 'CUDA RTX 4050' : 'CPU Engine'}
            </span>
          </div>
          <span style={{ color: 'var(--border-medium)' }}>|</span>
          <span>PyTorch {health?.torch_version || '2.14'}</span>
          <span style={{ color: 'var(--border-medium)' }}>|</span>
          <span>RDKit {health?.rdkit_version || '2026.09'}</span>
          <span style={{ color: 'var(--border-medium)' }}>|</span>
          <span style={{ color: 'var(--olive-muted)', fontWeight: 600 }}>Seed: 42 Deterministic</span>
        </div>
      </div>

      {/* Navigation tabs */}
      <div style={{
        maxWidth: 1300,
        margin: '12px auto 0 auto',
        display: 'flex',
        gap: 6,
      }}>
        {[
          { id: 'reaction-lab', label: 'Reaction Prediction Lab', icon: <FlaskConical size={15} /> },
          { id: 'baselines', label: 'Classical ML Baselines', icon: <GitBranch size={15} /> },
          { id: 'dataset', label: 'Dataset Explorer (900k+ USPTO)', icon: <Database size={15} /> },
          { id: 'hardware', label: 'Hardware Acceleration & VRAM', icon: <Cpu size={15} /> },
          { id: 'architecture', label: 'Architecture & Scratch Models', icon: <GitBranch size={15} /> },
        ].map(tab => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '8px 16px',
                fontSize: 13,
                fontWeight: isActive ? 600 : 500,
                color: isActive ? 'var(--sage-dark)' : 'var(--text-secondary)',
                backgroundColor: isActive ? 'var(--sage-light)' : 'transparent',
                border: isActive ? '1px solid var(--border-soft)' : '1px solid transparent',
                borderRadius: 6,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {tab.icon}
              {tab.label}
            </button>
          );
        })}
      </div>
    </header>
  );
};
