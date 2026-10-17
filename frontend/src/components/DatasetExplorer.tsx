import React, { useEffect, useState } from 'react';
import { Database, FileText, HardDrive, Layers, RefreshCw } from 'lucide-react';
import { fetchDatasetSummary } from '../api';
import type { DatasetSummary } from '../types';

export const DatasetExplorer: React.FC = () => {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const loadData = async () => {
    setLoading(true);
    const data = await fetchDatasetSummary();
    setSummary(data);
    setLoading(false);
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Overview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        <div style={{
          backgroundColor: '#FFFFFF',
          padding: '18px 20px',
          borderRadius: 8,
          border: '1px solid var(--border-soft)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 12 }}>
            <Layers size={16} />
            <span>Total Dataset Records</span>
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-charcoal)', marginTop: 6, fontFamily: 'serif' }}>
            910,000+
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
            US Patents 1976-Sep2016 + Jin USPTO
          </div>
        </div>

        <div style={{
          backgroundColor: '#FFFFFF',
          padding: '18px 20px',
          borderRadius: 8,
          border: '1px solid var(--border-soft)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 12 }}>
            <HardDrive size={16} />
            <span>Total Volume</span>
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-charcoal)', marginTop: 6, fontFamily: 'serif' }}>
            {summary?.total_size_mb ? `${summary.total_size_mb} MB` : '964.4 MB'}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
            6 raw splits & benchmark text files
          </div>
        </div>

        <div style={{
          backgroundColor: '#FFFFFF',
          padding: '18px 20px',
          borderRadius: 8,
          border: '1px solid var(--border-soft)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 12 }}>
            <Database size={16} />
            <span>Primary Training Split</span>
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-charcoal)', marginTop: 6, fontFamily: 'serif' }}>
            674.0 MB
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
            US_patents..._train.csv
          </div>
        </div>

        <div style={{
          backgroundColor: '#FFFFFF',
          padding: '18px 20px',
          borderRadius: 8,
          border: '1px solid var(--border-soft)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 12 }}>
            <FileText size={16} />
            <span>Streaming Strategy</span>
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--sage-dark)', marginTop: 6, fontFamily: 'serif' }}>
            Chunked Stream
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
            10,000 line chunks (RAM-bounded)
          </div>
        </div>
      </div>

      {/* Dataset Table Card */}
      <div style={{
        backgroundColor: '#FFFFFF',
        borderRadius: 10,
        border: '1px solid var(--border-soft)',
        padding: 24,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-charcoal)', fontFamily: 'serif' }}>
              Dataset Inventory & Split Specifications
            </h3>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Pre-staged datasets in directory <code>ReactionSeq2Seq_Dataset/</code>
            </p>
          </div>

          <button
            onClick={loadData}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 12px',
              borderRadius: 6,
              border: '1px solid var(--border-soft)',
              backgroundColor: 'var(--bg-subtle)',
              color: 'var(--text-secondary)',
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>

        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-medium)', textAlign: 'left', color: 'var(--text-muted)' }}>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Filename</th>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Format</th>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Size</th>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Research Role</th>
            </tr>
          </thead>
          <tbody>
            {summary?.files.map((file, i) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--border-soft)' }}>
                <td style={{ padding: '12px', fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-charcoal)' }}>
                  {file.filename}
                </td>
                <td style={{ padding: '12px' }}>
                  <span style={{
                    padding: '2px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                    fontWeight: 600,
                    backgroundColor: 'var(--bg-subtle)',
                    color: 'var(--text-secondary)',
                    border: '1px solid var(--border-soft)',
                  }}>
                    {file.format}
                  </span>
                </td>
                <td style={{ padding: '12px', fontWeight: 600, color: 'var(--text-charcoal)' }}>
                  {file.size_mb} MB
                </td>
                <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>
                  {file.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Architecture note for 900k scale */}
      <div style={{
        backgroundColor: 'var(--bg-subtle)',
        border: '1px solid var(--border-soft)',
        borderRadius: 8,
        padding: 20,
        fontSize: 13,
        color: 'var(--text-secondary)',
        lineHeight: 1.6,
      }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-charcoal)', marginBottom: 6, fontFamily: 'serif' }}>
          Memory-Bounded Pipeline Design for Laptop Workstations
        </h4>
        <p>
          Loading 900,000+ chemical reactions at once can exceed system RAM on laptops with 16GB memory. The ChemRxn-Intelligence <code>ReactionDataset</code> utilizes streaming chunked iterators (default chunk size 5,000–10,000 lines) and Windows-safe PyTorch <code>DataLoader</code> configurations. This guarantees smooth iteration, zero thread deadlock, and optimal GPU cache utilization on NVIDIA RTX 3050 and 4050 cards.
        </p>
      </div>
    </div>
  );
};
