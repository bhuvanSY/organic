import React from 'react';
import type { ActiveRoute, HealthInfo, HardwareInfo } from '../types';
import { 
  FlaskConical, 
  BarChart3, 
  Database, 
  BookOpen, 
  Compass, 
  Cpu, 
  Layers,
  Box
} from 'lucide-react';

interface NavbarProps {
  activeRoute: ActiveRoute;
  onRouteChange: (route: ActiveRoute) => void;
  health: HealthInfo | null;
  hardware: HardwareInfo | null;
  candidateCount?: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeRoute,
  onRouteChange,
  health,
  hardware,
  candidateCount = 0,
}) => {
  const isOnline = health?.status === 'ok';

  const navItems: { id: ActiveRoute; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'landing', label: 'Overview', icon: <Compass size={16} /> },
    { id: 'workbench', label: 'Reaction Workbench', icon: <FlaskConical size={16} /> },
    { 
      id: 'results', 
      label: 'Product Results', 
      icon: <Layers size={16} />,
      badge: candidateCount > 0 ? `${candidateCount}` : undefined
    },
    { id: 'explorer', label: '3D Explorer', icon: <Box size={16} /> },
    { id: 'benchmark', label: 'Benchmark Studio', icon: <BarChart3 size={16} /> },
    { id: 'dataset', label: 'Dataset & Splits', icon: <Database size={16} /> },
    { id: 'docs', label: 'Documentation', icon: <BookOpen size={16} /> },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#F7F7F2]/90 backdrop-blur-md border-b border-[#E5E6DE]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Scientific Masthead */}
          <div 
            onClick={() => onRouteChange('landing')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-9 h-9 rounded bg-[#87966A] flex items-center justify-center text-white font-mono font-bold text-sm shadow-sm transition-transform group-hover:scale-105">
              Rx
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-['Manrope'] font-extrabold text-[17px] tracking-tight text-[#252820]">
                  ChemRxn
                </span>
                <span className="text-[11px] font-mono font-semibold tracking-wider px-1.5 py-0.5 rounded bg-[#F0F0EA] text-[#5E6356] border border-[#E5E6DE]">
                  INTELLIGENCE
                </span>
              </div>
              <p className="text-[11px] text-[#898E81] tracking-wide font-medium">
                Organic Reaction Prediction Platform
              </p>
            </div>
          </div>

          {/* Editorial Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1" aria-label="Main Navigation">
            {navItems.map((item) => {
              const isActive = activeRoute === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onRouteChange(item.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-md text-[13px] font-medium transition-all ${
                    isActive
                      ? 'bg-white text-[#252820] shadow-sm border border-[#E5E6DE] font-semibold'
                      : 'text-[#5E6356] hover:text-[#252820] hover:bg-[#F0F0EA]'
                  }`}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <span className={isActive ? 'text-[#87966A]' : 'text-[#898E81]'}>
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[10px] font-mono font-bold bg-[#87966A] text-white">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Hardware & Runtime Telemetry Pill */}
          <div className="flex items-center gap-3">
            <div className="hidden lg:flex items-center gap-2 text-xs font-mono px-3 py-1.5 rounded-md bg-white border border-[#E5E6DE]">
              <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-[#4D7C0F]' : 'bg-[#B45309]'}`} />
              <span className="text-[#5E6356] font-medium">
                {isOnline ? (hardware?.device_name || 'CUDA Core') : 'Offline Telemetry'}
              </span>
              <span className="text-[#898E81] text-[11px]">|</span>
              <span className="text-[#898E81] text-[11px] flex items-center gap-1">
                <Cpu size={12} />
                {hardware?.device_type === 'cuda' ? 'FP16 Active' : 'CPU'}
              </span>
            </div>

            {/* Mobile Nav Trigger */}
            <div className="flex md:hidden">
              <select
                value={activeRoute}
                onChange={(e) => onRouteChange(e.target.value as ActiveRoute)}
                className="bg-white border border-[#E5E6DE] rounded-md px-3 py-1.5 text-xs font-medium text-[#252820]"
                aria-label="Select section"
              >
                {navItems.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

        </div>
      </div>
    </header>
  );
};
