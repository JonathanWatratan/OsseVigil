import { FileImage, UploadCloud, X } from 'lucide-react'
import { useRef, useState, type ChangeEvent, type DragEvent } from 'react'

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png'])
const MAX_SIZE = 25 * 1024 * 1024

interface UploadPanelProps {
  file: File | null
  disabled: boolean
  onFile: (file: File | null) => void
}

export function UploadPanel({ file, disabled, onFile }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')

  const accept = (candidate?: File) => {
    if (!candidate) return
    if (!ALLOWED_TYPES.has(candidate.type)) {
      setError('Please choose a JPEG or PNG image.')
      return
    }
    if (candidate.size > MAX_SIZE) {
      setError('Image exceeds the 25 MB upload limit.')
      return
    }
    setError('')
    onFile(candidate)
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    if (!disabled) accept(event.dataTransfer.files[0])
  }

  const onChange = (event: ChangeEvent<HTMLInputElement>) => accept(event.target.files?.[0])

  if (file) {
    return (
      <div className="selected-file-card">
        <span className="selected-file-icon"><FileImage size={24} /></span>
        <div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · {file.type}</span></div>
        <button className="icon-button" onClick={() => onFile(null)} disabled={disabled} aria-label="Remove selected image"><X size={18} /></button>
      </div>
    )
  }

  return (
    <div>
      <div
        className={`upload-dropzone ${dragging ? 'is-dragging' : ''} ${disabled ? 'is-disabled' : ''}`}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => (event.key === 'Enter' || event.key === ' ') && !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
      >
        <input ref={inputRef} type="file" accept="image/jpeg,image/png" onChange={onChange} hidden />
        <span className="upload-icon"><UploadCloud size={31} /></span>
        <strong>Drop a bone X-ray here</strong>
        <span>or click to choose a file</span>
        <small>JPEG or PNG · maximum 25 MB · one image per analysis</small>
      </div>
      {error && <p className="field-error" role="alert">{error}</p>}
    </div>
  )
}
