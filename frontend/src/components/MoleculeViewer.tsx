import React, { useState } from 'react';
import { getMoleculeSvgUrl } from '../api';
import { Copy, Check, AlertCircle } from 'lucide-react';

interface MoleculeViewerProps {
  smiles: string;
  name?: string;
  width?: number;
  height?: number;
  className?: string;
  showCopy?: boolean;
}

export const MoleculeViewer: React.FC<MoleculeViewerProps> = ({
  smiles,
  name,
  width = 300,
  height = 200,
  className = '',
  showCopy = true,
}) => {
  const [copied, setCopied] = useState(false);
  const [imgError, setImgError] = useState(false);

  const cleanSmiles = smiles ? smiles.trim() : '';
  const svgUrl = cleanSmiles ? getMoleculeSvgUrl(cleanSmiles, width, height) : '';

  const handleCopy = () => {
    if (!cleanSmiles) return;
    navigator.clipboard.writeText(cleanSmiles);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!cleanSmiles) {
    return (
      <div
        className={`flex flex-col items-center justify-center bg-[#F7F7F2] border border-[#E5E6DE] rounded-lg p-6 text-center ${className}`}
        style={{ width, height }}
      >
        <span className="text-xs text-[#898E81] font-mono">No structure provided</span>
      </div>
    );
  }

  return (
    <div
      className={`relative group bg-white border border-[#E5E6DE] rounded-lg overflow-hidden flex flex-col justify-between p-3 transition-all duration-150 hover:border-[#B6C49A] ${className}`}
      style={{ minWidth: width }}
    >
      {name && (
        <div className="flex items-center justify-between pb-2 border-b border-[#F0F0EA] mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#5E6356] font-mono">
            {name}
          </span>
          {showCopy && (
            <button
              onClick={handleCopy}
              className="text-[#898E81] hover:text-[#252820] text-xs flex items-center gap-1 transition-colors"
              title="Copy canonical SMILES"
              aria-label="Copy canonical SMILES"
            >
              {copied ? (
                <>
                  <Check size={13} className="text-[#4D7C0F]" />
                  <span className="text-[11px] text-[#4D7C0F]">Copied</span>
                </>
              ) : (
                <>
                  <Copy size={13} />
                  <span className="text-[11px]">SMILES</span>
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* Chemical Depiction Area */}
      <div className="flex-1 flex items-center justify-center min-h-[140px] bg-[#FAF9F5] rounded border border-[#F0F0EA] p-2 overflow-hidden">
        {!imgError ? (
          <img
            src={svgUrl}
            alt={`Chemical structure for ${cleanSmiles}`}
            className="max-w-full max-h-full object-contain filter drop-shadow-sm select-none"
            onError={() => setImgError(true)}
            loading="lazy"
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-center p-3">
            <AlertCircle size={18} className="text-[#87966A] mb-1" />
            <span className="text-[11px] font-semibold text-[#5E6356]">RDKit Depiction Unavailable</span>
            <span className="text-[10px] text-[#898E81] font-mono truncate max-w-[220px] mt-1">
              {cleanSmiles}
            </span>
          </div>
        )}
      </div>

      {/* SMILES text pill */}
      <div className="mt-2 pt-2 border-t border-[#F0F0EA] flex items-center justify-between gap-2">
        <code className="text-[11px] text-[#252820] font-mono truncate max-w-[240px]" title={cleanSmiles}>
          {cleanSmiles}
        </code>
      </div>
    </div>
  );
};
