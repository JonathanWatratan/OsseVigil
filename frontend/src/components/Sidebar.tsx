import { Activity, FileHeart, Info, Microscope, Plus, X } from 'lucide-react'
import type { AppPage } from '../types'
import { Logo } from './Logo'

interface SidebarProps {
  page: AppPage
  open: boolean
  onClose: () => void
  onNavigate: (page: AppPage) => void
  onNewAnalysis: () => void
}

export function Sidebar({ page, open, onClose, onNavigate, onNewAnalysis }: SidebarProps) {
  const navigate = (target: AppPage) => {
    onNavigate(target)
    onClose()
  }

  return (
    <>
      <button className={`mobile-scrim ${open ? 'is-visible' : ''}`} onClick={onClose} aria-label="Close navigation" />
      <aside className={`sidebar ${open ? 'is-open' : ''}`} aria-label="Primary navigation">
        <div className="sidebar-top">
          <Logo />
          <button className="icon-button sidebar-close" onClick={onClose} aria-label="Close navigation"><X size={20} /></button>
        </div>
        <button className="new-analysis-button" onClick={onNewAnalysis}>
          <Plus size={19} /> New Analysis
        </button>
        <nav className="sidebar-nav">
          <button className={page === 'workspace' ? 'is-active' : ''} onClick={() => navigate('workspace')}>
            <Activity size={19} /> <span>Analysis Workspace</span>
          </button>
          <button className={page === 'research' ? 'is-active' : ''} onClick={() => navigate('research')}>
            <Microscope size={19} /> <span>Research Information</span>
          </button>
          <button className={page === 'about' ? 'is-active' : ''} onClick={() => navigate('about')}>
            <Info size={19} /> <span>About OsteoInsight</span>
          </button>
        </nav>
        <div className="sidebar-disclaimer">
          <FileHeart size={18} />
          <p><strong>Research prototype</strong><br />Not validated for diagnosis or treatment decisions.</p>
        </div>
      </aside>
    </>
  )
}
