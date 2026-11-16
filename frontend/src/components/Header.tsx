import { FileText, Menu } from 'lucide-react'
import type { HealthResponse } from '../types'

interface HeaderProps {
  health: HealthResponse | null
  onMenu: () => void
  onCredits: () => void
}

export function Header({ health, onMenu, onCredits }: HeaderProps) {
  const ready = health?.status === 'ready'
  return (
    <header className="top-header">
      <button className="icon-button mobile-menu" onClick={onMenu} aria-label="Open navigation"><Menu size={21} /></button>
      <div className="top-header-title">
        <span className="eyebrow">OSTEOINSIGHT WORKSTATION</span>
        <span>Bone X-ray classification research</span>
      </div>
      <div className="top-header-actions">
        <span className={`status-pill ${ready ? 'is-ready' : 'is-unavailable'}`} title={health?.detail || 'Checking backend'}>
          <span className="status-dot" /> {ready ? 'Model ready' : health ? 'Model unavailable' : 'Connecting'}
        </span>
        <button className="credits-button" onClick={onCredits}>
          <FileText size={17} /> <span>License &amp; Credits</span>
        </button>
      </div>
    </header>
  )
}
