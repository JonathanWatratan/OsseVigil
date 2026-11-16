import { AlertTriangle, ArrowRight, Download, LoaderCircle, ScanSearch, ShieldAlert } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { OsteoInsightApiError, predictImage } from '../api'
import type { HealthResponse, PredictionResponse } from '../types'
import { UploadPanel } from './UploadPanel'
import { XrayViewer } from './XrayViewer'

interface AnalysisWorkspaceProps {
  health: HealthResponse | null
  resetToken: number
}

export function AnalysisWorkspace({ health, resetToken }: AnalysisWorkspaceProps) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [threshold, setThreshold] = useState(0.25)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setFile(null); setResult(null); setError(''); setLoading(false)
  }, [])

  useEffect(() => reset(), [resetToken, reset])
  useEffect(() => {
    if (!file) { setPreview(''); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const modelReady = health?.status === 'ready'
  const canAnalyze = Boolean(file && health && !loading)
  const statusText = useMemo(() => {
    if (loading) return 'Analyzing X-ray...'
    if (!health) return 'Connecting to localization service...'
    if (!modelReady) return 'Localization model unavailable'
    return 'BTXRD localization model ready'
  }, [health, loading, modelReady])

  const analyze = async () => {
    if (!file) return
    abortRef.current?.abort()
    const controller = new AbortController(); abortRef.current = controller
    setLoading(true); setError(''); setResult(null)
    try {
      setResult(await predictImage(file, threshold, controller.signal))
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      if (caught instanceof OsteoInsightApiError && caught.status === 503) {
        setError('The verified BTXRD detector is not available. No legacy-classifier fallback was used.')
      } else {
        setError(caught instanceof Error ? caught.message : 'The analysis could not be completed.')
      }
    } finally {
      setLoading(false)
    }
  }

  const downloadAnnotated = async () => {
    if (!preview || !result) return
    const image = new Image()
    image.src = preview
    await image.decode()
    const canvas = document.createElement('canvas')
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight
    const context = canvas.getContext('2d')
    if (!context) return
    context.drawImage(image, 0, 0)
    const stroke = Math.max(3, Math.round(canvas.width / 400))
    context.lineWidth = stroke
    context.font = `${Math.max(18, Math.round(canvas.width / 55))}px system-ui`
    result.detections.forEach((detection, index) => {
      const [x_min, y_min, x_max, y_max] = detection.bbox
      const x = x_min * canvas.width; const y = y_min * canvas.height
      const width = (x_max - x_min) * canvas.width; const height = (y_max - y_min) * canvas.height
      const centerX = x + width / 2; const centerY = y + height / 2
      context.strokeStyle = '#ff6b4a'
      context.lineWidth = stroke
      context.setLineDash([])
      context.beginPath()
      context.ellipse(centerX, centerY, width / 2, height / 2, 0, 0, Math.PI * 2)
      context.stroke()

      const fontSize = Math.max(18, Math.round(canvas.width / 55))
      context.font = `700 ${fontSize}px system-ui`
      const label = `Potential Bone Lesion ${index + 1}`
      const metrics = context.measureText(label)
      const labelHeight = Math.max(30, fontSize * 1.65)
      const labelWidth = metrics.width + 22
      const margin = Math.max(14, canvas.width * 0.015)
      const placeRight = x + width + labelWidth + margin < canvas.width
      const labelX = placeRight
        ? x + width + margin
        : Math.max(margin, x - labelWidth - margin)
      const labelY = Math.min(canvas.height - labelHeight - margin, Math.max(margin, y - labelHeight * 0.25))
      const lineStartX = placeRight ? x + width : x
      const lineEndX = placeRight ? labelX : labelX + labelWidth
      const lineEndY = labelY + labelHeight / 2

      context.strokeStyle = '#ff9f43'
      context.lineWidth = stroke
      context.beginPath(); context.moveTo(lineStartX, centerY); context.lineTo(lineEndX, lineEndY); context.stroke()
      const angle = Math.atan2(lineEndY - centerY, lineEndX - lineStartX)
      const arrow = Math.max(10, stroke * 4)
      context.fillStyle = '#ff9f43'
      context.beginPath()
      context.moveTo(lineEndX, lineEndY)
      context.lineTo(lineEndX - arrow * Math.cos(angle - Math.PI / 6), lineEndY - arrow * Math.sin(angle - Math.PI / 6))
      context.lineTo(lineEndX - arrow * Math.cos(angle + Math.PI / 6), lineEndY - arrow * Math.sin(angle + Math.PI / 6))
      context.closePath(); context.fill()

      context.fillStyle = '#071a2d'; context.fillRect(labelX, labelY, labelWidth, labelHeight)
      context.strokeStyle = '#ff9f43'; context.strokeRect(labelX, labelY, labelWidth, labelHeight)
      context.fillStyle = '#ffffff'
      context.fillText(label, labelX + 11, labelY + labelHeight * 0.68)
    })
    const link = document.createElement('a')
    link.download = `osteoinsight-annotated-${file?.name.replace(/\.[^.]+$/, '') || 'xray'}.png`
    link.href = canvas.toDataURL('image/png'); link.click()
  }

  return (
    <div className="workspace-page">
      <section className="page-intro">
        <span className="eyebrow">RESEARCH WORKSPACE</span>
        <h1>Bone Tumor X-Ray Analysis</h1>
        <p>Upload one bone radiograph to localize image regions that resemble BTXRD’s expert-annotated bone tumor lesions. Outputs require professional review.</p>
      </section>

      <div className="medical-banner"><ShieldAlert size={18} /><span><strong>Research prototype only.</strong> Not validated for diagnosis, staging, prognosis, surgery urgency, chemotherapy eligibility, or treatment decisions.</span></div>

      <section className="analysis-control-card">
        <div className="analysis-control-heading">
          <div><span className="step-number">01</span><div><h2>Upload radiograph</h2><p>Original files stay in your browser until you start analysis.</p></div></div>
          <span className={`inline-model-state ${modelReady ? 'is-ready' : ''}`}><span />{statusText}</span>
        </div>
        <UploadPanel file={file} disabled={loading} onFile={(next) => { setFile(next); setResult(null); setError('') }} />
        <div className="analysis-actions">
          <label className="threshold-control">
            <span><strong>Display threshold</strong><small>Uncalibrated detector score</small></span>
            <input type="range" min="0.05" max="0.95" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} disabled={loading} />
            <output>{threshold.toFixed(2)}</output>
          </label>
          <button className="primary-action" onClick={analyze} disabled={!canAnalyze}>
            {loading ? <><LoaderCircle className="spin" size={18} /> Analyzing X-ray...</> : <><ScanSearch size={18} /> Analyze X-ray <ArrowRight size={17} /></>}
          </button>
        </div>
        {!modelReady && health && <p className="model-unavailable-note"><AlertTriangle size={16} /> BTXRD checkpoint is not loaded. Analyze will contact the real API and report the unavailable state; no legacy or simulated prediction will be used.</p>}
        {error && <p className="analysis-error" role="alert"><AlertTriangle size={17} /> {error}</p>}
      </section>

      {preview && (
        <section className="comparison-section">
          <div className="section-heading"><div><span className="step-number">02</span><div><h2>Image comparison</h2><p>Overlays use actual normalized detector coordinates.</p></div></div>{result && <button className="secondary-action" onClick={downloadAnnotated}><Download size={17} /> Download annotated PNG</button>}</div>
          <div className="viewer-grid">
            <XrayViewer title="Original X-ray" imageUrl={preview} scanning={loading} />
            <XrayViewer title="AI-analyzed X-ray" imageUrl={preview} detections={result?.detections} scanning={loading} />
          </div>
        </section>
      )}

      {result && (
        <section className="results-section" aria-live="polite">
          <div className="section-heading"><div><span className="step-number">03</span><div><h2>Model results</h2><p>{result.model_version} · {result.inference_ms.toFixed(0)} ms model call</p></div></div></div>
          <div className={`result-summary ${result.detections.length ? 'has-detections' : 'is-empty'}`}>
            <ScanSearch size={22} />
            <div>
              <strong>{result.detections.length ? 'Potential lesion — requires professional review' : 'No lesion detected by this model.'}</strong>
              <p>{result.detections.length ? `${result.detections.length} model detection${result.detections.length > 1 ? 's' : ''} above threshold ${result.operating_threshold.toFixed(2)}.` : 'This does not rule out a lesion or cancer. Professional assessment may still be necessary.'}</p>
            </div>
          </div>
          {result.detections.length > 0 && <div className="detection-list">{result.detections.map((detection, index) => <article key={`${detection.bbox[0]}-${index}`}><span>{index + 1}</span><div><strong>Potential Bone Lesion</strong><small>Normalized box: [{detection.bbox.map((value) => value.toFixed(3)).join(', ')}]</small></div><output>{detection.score.toFixed(2)}<small>detector score</small></output></article>)}</div>}
          <div className="warning-list">{result.warnings.map((warning) => <p key={warning}><AlertTriangle size={15} />{warning}</p>)}</div>
        </section>
      )}
    </div>
  )
}
