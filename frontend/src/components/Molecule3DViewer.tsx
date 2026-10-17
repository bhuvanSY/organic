import React, { useEffect, useRef, useState } from 'react';
import type { StructureResponse, Atom3DCoord } from '../types';
import { fetchMoleculeStructure } from '../api';
import { getFallbackConformer } from '../conformerFallbacks';
import { MoleculeViewer } from './MoleculeViewer';
import { 
  RotateCw, 
  ZoomIn, 
  ZoomOut, 
  Maximize2, 
  Download, 
  Copy, 
  Check, 
  Tag, 
  AlertCircle
} from 'lucide-react';

interface Molecule3DViewerProps {
  smiles: string;
  name?: string;
  height?: number;
  highlightAtomIndices?: number[];
  showControls?: boolean;
}

type Representation = 'ballAndStick' | 'stick' | 'sphere';

export const Molecule3DViewer: React.FC<Molecule3DViewerProps> = ({
  smiles,
  name,
  height = 360,
  highlightAtomIndices = [],
  showControls = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);

  const [structure, setStructure] = useState<StructureResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [representation, setRepresentation] = useState<Representation>('ballAndStick');
  const [showLabels, setShowLabels] = useState<boolean>(false);
  const [includeHydrogens, setIncludeHydrogens] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [selectedAtom, setSelectedAtom] = useState<Atom3DCoord | null>(null);

  // 1. Fetch 3D conformer coordinates from RDKit backend
  useEffect(() => {
    let isMounted = true;
    if (!smiles || !smiles.trim()) {
      setStructure(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    setSelectedAtom(null);

    fetchMoleculeStructure(smiles, includeHydrogens, true)
      .then((res) => {
        if (!isMounted) return;
        if (res.success && res.mol_block) {
          setStructure(res);
        } else {
          const fallback = getFallbackConformer(smiles);
          if (fallback) {
            setStructure(fallback);
          } else {
            setError(res.error || 'Failed to generate 3D conformer');
          }
        }
      })
      .catch((err) => {
        if (!isMounted) return;
        const fallback = getFallbackConformer(smiles);
        if (fallback) {
          setStructure(fallback);
          setError(null);
        } else {
          setError(err.message || 'Conformer service unreachable');
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [smiles, includeHydrogens]);

  // 2. Initialize and render 3Dmol.js viewer
  useEffect(() => {
    if (!containerRef.current || !structure?.mol_block) return;

    const $3Dmol = (window as any).$3Dmol;
    if (!$3Dmol) {
      setError('3Dmol.js visualization library not loaded.');
      return;
    }

    // Clean up existing instance
    if (viewerRef.current) {
      try {
        viewerRef.current.clear();
      } catch {
        // no-op
      }
    }
    containerRef.current.innerHTML = '';

    const viewer = $3Dmol.createViewer(containerRef.current, {
      backgroundColor: '#FAF9F5',
      antialias: true,
    });
    viewerRef.current = viewer;

    // Add V2000 MOL Block model
    viewer.addModel(structure.mol_block, 'mol');

    // Apply representation
    applyRepresentation(viewer, representation, highlightAtomIndices);

    // Apply atom click listener for inspector
    viewer.setClickable({}, true, (atom: any) => {
      if (atom && structure.atoms && structure.atoms[atom.index]) {
        setSelectedAtom(structure.atoms[atom.index]);
      }
    });

    // Apply labels if enabled
    if (showLabels) {
      viewer.addPropertyLabels('elem', {}, {
        fontSize: 10,
        fontColor: '#252820',
        backgroundOpacity: 0.6,
        backgroundColor: '#FFFFFF',
      });
    }

    viewer.zoomTo();
    viewer.render();

    return () => {
      if (viewerRef.current) {
        try {
          viewerRef.current.clear();
        } catch {
          // no-op
        }
      }
    };
  }, [structure, representation, showLabels, highlightAtomIndices]);

  // Helper to configure styles
  const applyRepresentation = (viewer: any, rep: Representation, highlights: number[]) => {
    if (!viewer) return;

    if (rep === 'ballAndStick') {
      viewer.setStyle({}, { stick: { radius: 0.14, colorscheme: 'Jmol' }, sphere: { scale: 0.28, colorscheme: 'Jmol' } });
    } else if (rep === 'stick') {
      viewer.setStyle({}, { stick: { radius: 0.2, colorscheme: 'Jmol' } });
    } else if (rep === 'sphere') {
      viewer.setStyle({}, { sphere: { scale: 0.85, colorscheme: 'Jmol' } });
    }

    // Highlight reaction center atoms if present
    if (highlights && highlights.length > 0) {
      viewer.setStyle(
        { index: highlights },
        { 
          stick: { radius: 0.22, color: '#87966A' }, 
          sphere: { scale: 0.42, color: '#87966A' } 
        }
      );
    }
  };

  const handleResetCamera = () => {
    if (viewerRef.current) {
      viewerRef.current.zoomTo();
      viewerRef.current.render();
    }
  };

  const handleZoom = (factor: number) => {
    if (viewerRef.current) {
      viewerRef.current.zoom(factor);
      viewerRef.current.render();
    }
  };

  const handleExportSDF = () => {
    if (!structure?.sdf) return;
    const blob = new Blob([structure.sdf], { type: 'chemical/x-mdl-sdfile' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name || 'molecule'}_${Date.now()}.sdf`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopySMILES = () => {
    if (!smiles) return;
    navigator.clipboard.writeText(smiles);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white border border-[#E5E6DE] rounded-lg overflow-hidden flex flex-col shadow-sm">
      
      {/* Viewer Header Toolbar */}
      <div className="p-3 border-b border-[#E5E6DE] bg-[#FAF9F5] flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#252820]">
            {name || '3D Molecular Conformer'}
          </span>
          {structure?.force_field && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white border border-[#E5E6DE] text-[#4D7C0F]">
              {structure.force_field}
            </span>
          )}
        </div>

        {showControls && (
          <div className="flex items-center gap-1.5 text-xs">
            {/* Representation Selector */}
            <div className="flex items-center rounded border border-[#E5E6DE] bg-white p-0.5">
              <button
                type="button"
                onClick={() => setRepresentation('ballAndStick')}
                className={`px-2 py-1 rounded text-[11px] font-mono ${
                  representation === 'ballAndStick' ? 'bg-[#87966A] text-white font-bold' : 'text-[#5E6356] hover:text-[#252820]'
                }`}
                title="Ball & Stick Representation"
              >
                Ball-Stick
              </button>
              <button
                type="button"
                onClick={() => setRepresentation('stick')}
                className={`px-2 py-1 rounded text-[11px] font-mono ${
                  representation === 'stick' ? 'bg-[#87966A] text-white font-bold' : 'text-[#5E6356] hover:text-[#252820]'
                }`}
                title="Stick Representation"
              >
                Stick
              </button>
              <button
                type="button"
                onClick={() => setRepresentation('sphere')}
                className={`px-2 py-1 rounded text-[11px] font-mono ${
                  representation === 'sphere' ? 'bg-[#87966A] text-white font-bold' : 'text-[#5E6356] hover:text-[#252820]'
                }`}
                title="Van der Waals Spheres"
              >
                Spheres
              </button>
            </div>

            {/* Toggle Labels */}
            <button
              type="button"
              onClick={() => setShowLabels(!showLabels)}
              className={`p-1.5 rounded border border-[#E5E6DE] ${
                showLabels ? 'bg-[#87966A] text-white' : 'bg-white text-[#5E6356] hover:text-[#252820]'
              }`}
              title="Toggle Atom Element Labels"
            >
              <Tag size={13} />
            </button>

            {/* Toggle Hydrogens */}
            <button
              type="button"
              onClick={() => setIncludeHydrogens(!includeHydrogens)}
              className={`px-2 py-1 rounded border border-[#E5E6DE] text-[11px] font-mono ${
                includeHydrogens ? 'bg-[#87966A] text-white' : 'bg-white text-[#5E6356] hover:text-[#252820]'
              }`}
              title="Toggle Explicit Hydrogens"
            >
              {includeHydrogens ? '+H' : '-H'}
            </button>

            {/* Reset Camera */}
            <button
              type="button"
              onClick={handleResetCamera}
              className="p-1.5 rounded border border-[#E5E6DE] bg-white text-[#5E6356] hover:text-[#252820]"
              title="Reset Viewpoint"
            >
              <Maximize2 size={13} />
            </button>

            {/* Zoom In/Out */}
            <button
              type="button"
              onClick={() => handleZoom(1.2)}
              className="p-1.5 rounded border border-[#E5E6DE] bg-white text-[#5E6356] hover:text-[#252820]"
              title="Zoom In"
            >
              <ZoomIn size={13} />
            </button>
            <button
              type="button"
              onClick={() => handleZoom(0.8)}
              className="p-1.5 rounded border border-[#E5E6DE] bg-white text-[#5E6356] hover:text-[#252820]"
              title="Zoom Out"
            >
              <ZoomOut size={13} />
            </button>

            {/* Export SDF */}
            <button
              type="button"
              onClick={handleExportSDF}
              disabled={!structure?.sdf}
              className="px-2 py-1 rounded border border-[#E5E6DE] bg-white text-[#5E6356] hover:text-[#252820] text-[11px] font-mono flex items-center gap-1"
              title="Download 3D SDF File"
            >
              <Download size={12} />
              <span>SDF</span>
            </button>
          </div>
        )}
      </div>

      {/* Main 3D Canvas / Fallback Area */}
      <div className="relative w-full overflow-hidden bg-[#FAF9F5]" style={{ height }}>
        {loading ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#FAF9F5]/90 z-10 text-xs font-mono text-[#5E6356] space-y-2">
            <RotateCw size={20} className="animate-spin text-[#87966A]" />
            <span>Computing ETKDGv3 Conformer & MMFF94 Minimization...</span>
          </div>
        ) : error ? (
          <div className="p-6 h-full flex flex-col items-center justify-center text-center space-y-3">
            <AlertCircle size={24} className="text-[#B91C1C]" />
            <div className="text-xs font-bold text-[#252820]">3D Conformer Generation Unavailable</div>
            <p className="text-[11px] text-[#5E6356] max-w-sm">{error}</p>
            <span className="text-[10px] font-mono text-[#898E81]">Rendering 2D Vector Fallback:</span>
            <div className="bg-white p-2 rounded border border-[#E5E6DE] max-w-xs">
              <MoleculeViewer smiles={smiles} width={260} height={140} showCopy={false} />
            </div>
          </div>
        ) : (
          <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
        )}

        {/* Selected Atom Inspector Overlay */}
        {selectedAtom && (
          <div className="absolute bottom-3 left-3 bg-white/95 backdrop-blur-sm border border-[#E5E6DE] rounded-md p-3 shadow-md text-xs space-y-1 z-20 max-w-xs">
            <div className="flex items-center justify-between border-b border-[#F0F0EA] pb-1">
              <span className="font-bold text-[#252820] font-mono">
                Atom #{selectedAtom.index} ({selectedAtom.symbol})
              </span>
              <button
                type="button"
                onClick={() => setSelectedAtom(null)}
                className="text-[#898E81] hover:text-[#252820] text-xs font-bold"
              >
                ✕
              </button>
            </div>
            <div className="grid grid-cols-2 gap-x-3 text-[11px] font-mono text-[#5E6356] pt-1">
              <span>Hybridization:</span>
              <span className="font-semibold text-[#252820]">{selectedAtom.hybridization}</span>
              <span>Coordinates:</span>
              <span className="font-semibold text-[#252820]">
                ({selectedAtom.x.toFixed(2)}, {selectedAtom.y.toFixed(2)}, {selectedAtom.z.toFixed(2)})
              </span>
              <span>Formal Charge:</span>
              <span className="font-semibold text-[#252820]">{selectedAtom.formal_charge}</span>
              <span>Aromatic:</span>
              <span className="font-semibold text-[#252820]">{selectedAtom.is_aromatic ? 'Yes' : 'No'}</span>
            </div>
          </div>
        )}
      </div>

      {/* Footer Meta & Interaction Guide */}
      <div className="p-2.5 border-t border-[#E5E6DE] bg-white flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#898E81] font-mono">
        <div className="flex items-center gap-3">
          <span>Rotate: Left Click + Drag</span>
          <span>•</span>
          <span>Pan: Right Click + Drag</span>
          <span>•</span>
          <span>Inspect: Click Atom</span>
        </div>

        <div className="flex items-center gap-2">
          {structure?.energy_kcal_mol !== null && structure?.energy_kcal_mol !== undefined && (
            <span className="text-[#4D7C0F]">
              E = {structure.energy_kcal_mol} kcal/mol
            </span>
          )}
          <button
            type="button"
            onClick={handleCopySMILES}
            className="hover:text-[#252820] flex items-center gap-1 transition-colors"
            title="Copy SMILES"
          >
            {copied ? <Check size={12} className="text-[#4D7C0F]" /> : <Copy size={12} />}
            <span>{copied ? 'Copied' : 'SMILES'}</span>
          </button>
        </div>
      </div>

    </div>
  );
};
