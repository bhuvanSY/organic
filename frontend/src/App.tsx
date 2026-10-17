import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { LandingView } from './components/LandingView';
import { WorkbenchView } from './components/WorkbenchView';
import { ProductResultsView } from './components/ProductResultsView';
import { BenchmarkStudio } from './components/BenchmarkStudio';
import { DatasetExperimentView } from './components/DatasetExperimentView';
import { DocumentationView } from './components/DocumentationView';
import { MolecularExplorerView } from './components/MolecularExplorerView';

import { fetchHardware, fetchHealth } from './api';
import type { ActiveRoute, HardwareInfo, HealthInfo, PredictionResult } from './types';

export const App: React.FC = () => {
  const [activeRoute, setActiveRoute] = useState<ActiveRoute>('landing');
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);

  useEffect(() => {
    const init = async () => {
      const [h, hw] = await Promise.all([fetchHealth(), fetchHardware()]);
      setHealth(h);
      setHardware(hw);
    };
    init();
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#F7F7F2] text-[#252820]">
      {/* Top Editorial Navbar */}
      <Navbar
        activeRoute={activeRoute}
        onRouteChange={setActiveRoute}
        health={health}
        hardware={hardware}
        candidateCount={currentPrediction?.candidates?.length || 0}
      />

      {/* Main Content Router */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        {activeRoute === 'landing' && (
          <LandingView onNavigate={setActiveRoute} />
        )}

        {activeRoute === 'workbench' && (
          <WorkbenchView
            onPredictionComplete={(result) => {
              setCurrentPrediction(result);
              setActiveRoute('results');
            }}
            onNavigate={setActiveRoute}
          />
        )}

        {activeRoute === 'results' && (
          <ProductResultsView
            result={currentPrediction}
            onNavigate={setActiveRoute}
          />
        )}

        {activeRoute === 'explorer' && (
          <MolecularExplorerView
            currentPrediction={currentPrediction}
          />
        )}

        {activeRoute === 'benchmark' && (
          <BenchmarkStudio />
        )}

        {activeRoute === 'dataset' && (
          <DatasetExperimentView />
        )}

        {activeRoute === 'docs' && (
          <DocumentationView />
        )}
      </main>

      {/* Scientific Editorial Footer */}
      <footer className="mt-auto border-t border-[#E5E6DE] bg-white py-8 text-xs text-[#5E6356]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center sm:text-left">
            <div className="font-semibold text-[#252820] font-['Manrope']">
              ChemRxn-Intelligence — Forward Organic Chemical Prediction Platform
            </div>
            <p className="text-[#898E81] text-[11px]">
              Multi-modal synthesis modeling with scratch Transformers, Graph Neural Networks, and Platt calibration.
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-4 text-[11px] font-mono text-[#898E81]">
            <span>RDKit 2026.09 Kernel</span>
            <span>•</span>
            <span>CUDA Acceleration Active</span>
            <span>•</span>
            <span>Seed 42 Deterministic</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
