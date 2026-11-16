import { AlertTriangle, ArrowRight, BrainCircuit, LoaderCircle, ShieldAlert } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { OsteoInsightApiError, predictImage } from '../api'
import type { HealthResponse, PredictionResponse } from '../types'
import { UploadPanel } from './UploadPanel'

interface ClassificationWorkspaceProps {
  health: HealthResponse | null
  resetToken: number
}

export function ClassificationWorkspace({ health, resetToken }: ClassificationWorkspaceProps) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setFile(null)
    setResult(null)
    setError('')
    setLoading(false)
  }, [])

  useEffect(() => reset(), [resetToken, reset])
  useEffect(() => {
    if (!file) { setPreview(''); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const modelReady = health?.status === 'ready'
  const canAnalyze = Boolean(file && modelReady && !loading)
  const statusText = useMemo(() => {
    if (loading) return 'Analyzing X-ray...'
    if (!health) return 'Connecting to classification service...'
    if (!modelReady) return 'Classification model unavailable'
    return 'ResNet18 classification model ready'
  }, [health, loading, modelReady])

  const analyze = async () => {
    if (!file || !modelReady) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(true)
    setError('')
    setResult(null)
    try {
      setResult(await predictImage(file, controller.signal))
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      if (caught instanceof OsteoInsightApiError) setError(caught.message)
      else setError(caught instanceof Error ? caught.message : 'The analysis could not be completed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="workspace-page">
      <section className="page-intro">
        <span className="eyebrow">RESEARCH WORKSPACE</span>
        <h1>Bone X-Ray Classification</h1>
        <p>Compare one bone radiograph with the two categories in the project dataset. Scores describe model similarity, not clinical probability.</p>
      </section>

      <div className="medical-banner"><ShieldAlert size={18} /><span><strong>Research prototype only.</strong> Not validated for diagnosis, cancer exclusion, staging, prognosis, or treatment decisions.</span></div>

      <section className="analysis-control-card">
        <div className="analysis-control-heading">
          <div><span className="step-number">01</span><div><h2>Upload radiograph</h2><p>Choose one readable JPG or PNG image, up to 25 MB.</p></div></div>
          <span className={`inline-model-state ${modelReady ? 'is-ready' : ''}`}><span />{statusText}</span>
        </div>
        <UploadPanel file={file} disabled={loading} onFile={(next) => { setFile(next); setResult(null); setError('') }} />
        <div className="classification-action-row">
          <p><strong>Output:</strong> Cancer and Normal dataset-category scores</p>
          <button className="primary-action" onClick={analyze} disabled={!canAnalyze}>
            {loading ? <><LoaderCircle className="spin" size={18} /> Analyzing X-ray...</> : <><BrainCircuit size={18} /> Analyze X-ray <ArrowRight size={17} /></>}
          </button>
        </div>
        {!modelReady && health && <p className="model-unavailable-note"><AlertTriangle size={16} /> The ResNet18 checkpoint is not loaded. Check the backend health response and model path.</p>}
        {error && <p className="analysis-error" role="alert"><AlertTriangle size={17} /> {error}</p>}
      </section>

      {preview && (
        <section className="classification-grid" aria-live="polite">
          <article className="viewer-card classification-image-card">
            <div className="viewer-header"><div><span>Original X-ray</span><small>{file?.name}</small></div></div>
            <div className="classification-image-viewport"><img src={preview} alt="Uploaded bone X-ray" />{loading && <div className="scan-line" />}</div>
            <div className="viewer-footer">Original browser preview · no invented overlays</div>
          </article>

          <article className="score-panel">
            <div className="score-panel-heading">
              <div><span className="step-number">02</span><div><h2>Classification results</h2><p>{result ? `${result.model_version} · ${result.inference_ms.toFixed(0)} ms` : 'Run analysis to view real model output.'}</p></div></div>
            </div>
            {loading && <div className="score-placeholder"><LoaderCircle className="spin" size={25} /><strong>Running ResNet18 inference…</strong><span>The scores below will come from the loaded checkpoint.</span></div>}
            {!loading && !result && <div className="score-placeholder"><BrainCircuit size={25} /><strong>No result yet</strong><span>Upload an image and select Analyze X-ray.</span></div>}
            {result && <>
              <div className="research-result-label"><strong>Closest dataset category: {result.predicted_display_name}</strong><span>Inconclusive for clinical use — specialist review required</span></div>
              <div className="category-scores">
                {result.scores.map((item) => <div className="category-score" key={item.category}>
                  <div><strong>{item.display_name}</strong><output>{item.percentage.toFixed(1)}%</output></div>
                  <div className="score-track" aria-label={`${item.display_name} model category score ${item.percentage.toFixed(1)} percent`}><span style={{ width: `${item.percentage}%` }} /></div>
                </div>)}
              </div>
              <p className="score-meaning"><strong>Model category score — not the probability of cancer.</strong> {result.interpretation}</p>
              <div className="warning-list">{result.warnings.map((warning) => <p key={warning}><AlertTriangle size={15} />{warning}</p>)}</div>
            </>}
          </article>
        </section>
      )}
    </div>
  )
}
