import React, { useEffect, useState } from 'react';
import { Check, Cpu, RefreshCw, Zap } from 'lucide-react';
import { fetchHardware } from '../api';
import type { HardwareInfo } from '../types';

export const HardwareMonitor: React.FC = () => {
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const loadData = async () => {
    setLoading(true);
    const data = await fetchHardware();
    setHw(data);
    setLoading(false);
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Hardware Telemetry Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 24 }}>
        {/* GPU Details */}
        <div style={{
          backgroundColor: '#FFFFFF',
          borderRadius: 10,
          border: '1px solid var(--border-soft)',
          padding: 24,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Cpu size={20} color="var(--sage-base)" />
              <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-charcoal)', fontFamily: 'serif' }}>
                Primary Compute Acceleration Engine
              </h3>
            </div>
            <button
              onClick={loadData}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 10px',
                borderRadius: 6,
                border: '1px solid var(--border-soft)',
                backgroundColor: 'var(--bg-subtle)',
                color: 'var(--text-secondary)',
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
              Poll
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-soft)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Device Name:</span>
              <strong style={{ color: 'var(--text-charcoal)' }}>{hw?.device_name || 'NVIDIA GeForce RTX'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-soft)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Device Type:</span>
              <span style={{
                padding: '2px 8px',
                borderRadius: 4,
                backgroundColor: 'var(--sage-light)',
                color: 'var(--sage-dark)',
                fontWeight: 600,
                fontSize: 11,
              }}>
                {hw?.device_type.toUpperCase()}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-soft)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Dedicated Video Memory (VRAM):</span>
              <strong style={{ color: 'var(--text-charcoal)' }}>{hw?.total_memory_gb} GB</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-soft)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>CUDA Compute Capability:</span>
              <strong style={{ color: 'var(--text-charcoal)' }}>{hw?.compute_capability || '8.9'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-soft)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Streaming Multiprocessors (SM):</span>
              <strong style={{ color: 'var(--text-charcoal)' }}>{hw?.multi_processor_count || 20} SMs</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Precision Support:</span>
              <div style={{ display: 'flex', gap: 8 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--sage-dark)', backgroundColor: 'var(--sage-light)', padding: '2px 6px', borderRadius: 4 }}>
                  FP16 Mixed Precision
                </span>
                <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--sage-dark)', backgroundColor: 'var(--sage-light)', padding: '2px 6px', borderRadius: 4 }}>
                  BF16 Ready
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Windows Laptop Optimization Strategy */}
        <div style={{
          backgroundColor: '#FFFFFF',
          borderRadius: 10,
          border: '1px solid var(--border-soft)',
          padding: 24,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
            <Zap size={20} color="var(--amber-soft)" />
            <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-charcoal)', fontFamily: 'serif' }}>
              RTX 3050/4050 Laptop Optimizations
            </h3>
          </div>

          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 12, fontSize: 12, color: 'var(--text-secondary)' }}>
            <li style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
              <Check size={16} color="var(--sage-base)" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong style={{ color: 'var(--text-charcoal)' }}>Automatic Mixed Precision (AMP):</strong> Enables fp16 calculations on Tensor Cores, halving memory bandwidth and fitting seq2seq models easily in 4–6GB VRAM.
              </div>
            </li>
            <li style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
              <Check size={16} color="var(--sage-base)" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong style={{ color: 'var(--text-charcoal)' }}>Gradient Accumulation:</strong> Simulates large effective batch sizes (e.g. 128) using micro-batches of 32, preventing out-of-memory errors.
              </div>
            </li>
            <li style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
              <Check size={16} color="var(--sage-base)" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong style={{ color: 'var(--text-charcoal)' }}>Windows Multiprocessing Safety:</strong> Default <code>num_workers=0</code> to eliminate CUDA context recreation bugs and CPU-RAM duplication on Windows.
              </div>
            </li>
            <li style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
              <Check size={16} color="var(--sage-base)" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong style={{ color: 'var(--text-charcoal)' }}>CPU Fallback Guarantee:</strong> When CUDA is unavailable or memory exhausted, the system transparently runs on CPU without crashing.
              </div>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
