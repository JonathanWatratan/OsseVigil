import { Eye, EyeOff, Maximize2, Minus, Plus, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from 'react'
import type { Detection } from '../types'

interface XrayViewerProps {
  title: string
  imageUrl: string
  detections?: Detection[]
  scanning?: boolean
}

export function XrayViewer({ title, imageUrl, detections = [], scanning = false }: XrayViewerProps) {
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [overlays, setOverlays] = useState(true)
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 })
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 })
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)
  const hasOverlays = detections.length > 0

  const transform = useMemo(() => `translate(${offset.x}px, ${offset.y}px) scale(${scale})`, [offset, scale])
  const reset = () => { setScale(1); setOffset({ x: 0, y: 0 }) }
  const adjustScale = (next: number) => {
    const bounded = Math.min(4, Math.max(1, Number(next.toFixed(2))))
    setScale(bounded)
    if (bounded === 1) setOffset({ x: 0, y: 0 })
  }

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    const update = () => setViewportSize({ width: viewport.clientWidth, height: viewport.clientHeight })
    update()
    const observer = new ResizeObserver(update)
    observer.observe(viewport)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    reset()
    setNaturalSize({ width: 0, height: 0 })
  }, [imageUrl])

  const fittedSize = useMemo(() => {
    const { width: imageWidth, height: imageHeight } = naturalSize
    const { width: containerWidth, height: containerHeight } = viewportSize
    if (!imageWidth || !imageHeight || !containerWidth || !containerHeight) {
      return { width: containerWidth || 1, height: containerHeight || 1 }
    }
    const imageRatio = imageWidth / imageHeight
    const containerRatio = containerWidth / containerHeight
    return containerRatio > imageRatio
      ? { width: containerHeight * imageRatio, height: containerHeight }
      : { width: containerWidth, height: containerWidth / imageRatio }
  }, [naturalSize, viewportSize])

  const onWheel = (event: WheelEvent) => {
    event.preventDefault()
    adjustScale(scale + (event.deltaY < 0 ? 0.15 : -0.15))
  }
  const onPointerDown = (event: PointerEvent) => {
    if (scale <= 1) return
    dragRef.current = { x: event.clientX, y: event.clientY, ox: offset.x, oy: offset.y }
    event.currentTarget.setPointerCapture(event.pointerId)
  }
  const onPointerMove = (event: PointerEvent) => {
    if (!dragRef.current) return
    setOffset({ x: dragRef.current.ox + event.clientX - dragRef.current.x, y: dragRef.current.oy + event.clientY - dragRef.current.y })
  }
  const onPointerUp = () => { dragRef.current = null }

  return (
    <section className="viewer-card">
      <header className="viewer-header">
        <div><span>{title}</span><small>{Math.round(scale * 100)}%</small></div>
        <div className="viewer-tools" aria-label={`${title} controls`}>
          <button onClick={() => adjustScale(scale - 0.2)} aria-label="Zoom out"><Minus size={15} /></button>
          <button onClick={() => adjustScale(scale + 0.2)} aria-label="Zoom in"><Plus size={15} /></button>
          <button onClick={reset} aria-label="Reset zoom and pan"><RotateCcw size={15} /></button>
          {hasOverlays && <button onClick={() => setOverlays(!overlays)} aria-label={overlays ? 'Hide detections' : 'Show detections'}>{overlays ? <Eye size={15} /> : <EyeOff size={15} />}</button>}
        </div>
      </header>
      <div
        ref={viewportRef}
        className={`image-viewport ${scale > 1 ? 'is-zoomed' : ''}`}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <div className="image-stage">
          <div className="image-layer" style={{ ...fittedSize, transform }}>
            <img
              src={imageUrl}
              alt={title}
              draggable={false}
              onLoad={(event) => setNaturalSize({
                width: event.currentTarget.naturalWidth,
                height: event.currentTarget.naturalHeight,
              })}
            />
            {overlays && hasOverlays && naturalSize.width > 0 && naturalSize.height > 0 && (
              <svg className="detection-overlay" viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`} preserveAspectRatio="none" aria-label={`${detections.length} model detections`}>
                <defs>
                  <marker id="lesion-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" className="lesion-arrowhead" />
                  </marker>
                </defs>
                {detections.map((detection, index) => {
                  const [xMin, yMin, xMax, yMax] = detection.bbox
                  const x = xMin * naturalSize.width
                  const y = yMin * naturalSize.height
                  const width = (xMax - xMin) * naturalSize.width
                  const height = (yMax - yMin) * naturalSize.height
                  const centerX = x + width / 2
                  const centerY = y + height / 2
                  const fontSize = Math.max(13, naturalSize.width / 58)
                  const labelHeight = fontSize * 1.65
                  const labelWidth = fontSize * 12.8
                  const margin = naturalSize.width * 0.018
                  const placeRight = x + width + labelWidth + margin < naturalSize.width
                  const labelX = placeRight
                    ? x + width + margin
                    : Math.max(margin, x - labelWidth - margin)
                  const labelY = Math.min(
                    naturalSize.height - labelHeight - margin,
                    Math.max(margin, y - labelHeight * 0.25),
                  )
                  const connectorStartX = placeRight ? x + width : x
                  const connectorEndX = placeRight ? labelX : labelX + labelWidth
                  return (
                    <g key={`${xMin}-${yMin}-${index}`}>
                      <rect x={x} y={y} width={width} height={height} className="lesion-box" vectorEffect="non-scaling-stroke" />
                      <ellipse cx={centerX} cy={centerY} rx={width / 2} ry={height / 2} className="lesion-ellipse" vectorEffect="non-scaling-stroke" />
                      <line x1={connectorStartX} y1={centerY} x2={connectorEndX} y2={labelY + labelHeight / 2} className="lesion-connector" markerEnd="url(#lesion-arrow)" vectorEffect="non-scaling-stroke" />
                      <rect x={labelX} y={labelY} width={labelWidth} height={labelHeight} rx={fontSize * 0.28} className="lesion-label-bg" vectorEffect="non-scaling-stroke" />
                      <text x={labelX + fontSize * 0.55} y={labelY + labelHeight * 0.68} fontSize={fontSize} className="lesion-label-text">Potential Bone Lesion {index + 1}</text>
                    </g>
                  )
                })}
              </svg>
            )}
            {scanning && <span className="scan-line" aria-hidden="true" />}
          </div>
        </div>
      </div>
      <footer className="viewer-footer"><Maximize2 size={14} /> Scroll to zoom · drag to pan when zoomed</footer>
    </section>
  )
}
