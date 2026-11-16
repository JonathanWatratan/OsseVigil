import { AnimatePresence } from 'framer-motion'
import { useCallback, useEffect, useState } from 'react'
import { getHealth } from './api'
import { AboutPage } from './components/AboutPage'
import { ClassificationWorkspace } from './components/ClassificationWorkspace'
import { CreditsModal } from './components/CreditsModal'
import { Header } from './components/Header'
import { ResearchPage } from './components/ResearchPage'
import { Sidebar } from './components/Sidebar'
import { SplashScreen } from './components/SplashScreen'
import type { AppPage, HealthResponse } from './types'

const SPLASH_KEY = 'osteoinsight:splash-seen'

export default function App() {
  const [showSplash, setShowSplash] = useState(() => sessionStorage.getItem(SPLASH_KEY) !== 'true')
  const [page, setPage] = useState<AppPage>('workspace')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [creditsOpen, setCreditsOpen] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [resetToken, setResetToken] = useState(0)

  useEffect(() => {
    if (!showSplash) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const timer = window.setTimeout(() => {
      sessionStorage.setItem(SPLASH_KEY, 'true')
      setShowSplash(false)
    }, reduced ? 100 : 1500)
    return () => window.clearTimeout(timer)
  }, [showSplash])

  const refreshHealth = useCallback(async (signal?: AbortSignal) => {
    try {
      setHealth(await getHealth(signal))
    } catch {
      setHealth({
        status: 'model_not_ready', model_status: 'model_not_ready', model_version: 'unavailable',
        task: 'dataset_category_classification', checkpoint: '', device: '',
        detail: 'Backend service is unreachable', class_names: ['cancer', 'normal'],
      })
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    refreshHealth(controller.signal)
    const interval = window.setInterval(() => refreshHealth(), 30_000)
    return () => { controller.abort(); window.clearInterval(interval) }
  }, [refreshHealth])

  const newAnalysis = () => {
    setPage('workspace')
    setResetToken((value) => value + 1)
    setSidebarOpen(false)
  }

  return (
    <>
      <AnimatePresence>{showSplash && <SplashScreen />}</AnimatePresence>
      <div className="app-shell">
        <Sidebar page={page} open={sidebarOpen} onClose={() => setSidebarOpen(false)} onNavigate={setPage} onNewAnalysis={newAnalysis} />
        <div className="app-main">
          <Header health={health} onMenu={() => setSidebarOpen(true)} onCredits={() => setCreditsOpen(true)} />
          <main id="main-content">
            {page === 'workspace' && <ClassificationWorkspace health={health} resetToken={resetToken} />}
            {page === 'research' && <ResearchPage health={health} />}
            {page === 'about' && <AboutPage />}
          </main>
          <footer className="app-footer">Research prototype only · Not validated for clinical diagnosis, cancer staging, prognosis, or treatment decisions.</footer>
        </div>
      </div>
      <CreditsModal open={creditsOpen} activeDataset={health?.status === 'ready' ? 'Roboflow' : 'None'} onClose={() => setCreditsOpen(false)} />
    </>
  )
}
