import { ExternalLink, Info, X } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { technologies } from '../data/credits'

interface CreditsModalProps {
  open: boolean
  activeDataset: string
  onClose: () => void
}

export function CreditsModal({ open, activeDataset, onClose }: CreditsModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null)
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    closeRef.current?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (event.key !== 'Tab' || !dialogRef.current) return
      const focusable = Array.from(dialogRef.current.querySelectorAll<HTMLElement>('button, a[href], [tabindex]:not([tabindex="-1"])'))
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault(); last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      previous?.focus()
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="credits-modal" role="dialog" aria-modal="true" aria-labelledby="credits-title" ref={dialogRef}>
        <header className="credits-modal-header">
          <div className="credits-title-row">
            <span className="credits-icon"><Info size={24} /></span>
            <div>
              <h2 id="credits-title">OsteoInsight — License &amp; Credits</h2>
              <p>Research Project — Veritas AI International High School Hackathon 2026</p>
            </div>
          </div>
          <button className="modal-close" onClick={onClose} ref={closeRef} aria-label="Close License and Credits"><X size={20} /></button>
        </header>
        <div className="credits-modal-body">
          <section>
            <h3>Project Development Team</h3>
            <ol className="developer-list">
              <li>Jonathan Barack Watratan</li>
              <li>Rais Akbar Nasution</li>
              <li>Nicholas Azriel Manurung</li>
            </ol>
          </section>
          <section>
            <h3>School Affiliation</h3>
            <p><strong>SMA Kemala Taruna Bhayangkara</strong></p>
            <p>Kp. Tapos RT.01/02 serta RT.03/05, Desa Cibinong, Kecamatan Gunung Sindur, Kabupaten Bogor, Jawa Barat, Indonesia.</p>
            <p>NPSN: <a href="https://referensi.data.kemendikdasmen.go.id/pendidikan/npsn/70053796" target="_blank" rel="noreferrer">70053796 <ExternalLink size={13} /></a></p>
          </section>
          <section>
            <h3>Research &amp; Dataset References</h3>
            <div className="reference-card">
              <div><strong>BTXRD — Bone Tumor X-ray Radiograph Dataset</strong><span className="reference-state">Localization research archive</span></div>
              <p>Yao, S. et al. (2025). “A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors.” <em>Scientific Data</em>, 12, Article 88.</p>
              <a href="https://doi.org/10.1038/s41597-024-04311-y" target="_blank" rel="noreferrer">Research article <ExternalLink size={13} /></a>
              <a href="https://doi.org/10.6084/m9.figshare.27865398" target="_blank" rel="noreferrer">Original dataset distribution <ExternalLink size={13} /></a>
              <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">Dataset license: CC BY 4.0 <ExternalLink size={13} /></a>
            </div>
            <div className="reference-card active-reference">
              <div><strong>Roboflow Bone Cancer Detection v1</strong><span className="reference-state">{activeDataset === 'Roboflow' ? 'Active research model dataset' : 'Classifier not active'}</span></div>
              <p>The active ResNet18 reproduces two dataset categories. Its scores are not clinical cancer probabilities or diagnoses.</p>
              <a href="https://universe.roboflow.com/normal-bones/bone-cancer-detection-xa7ru/dataset/1" target="_blank" rel="noreferrer">Dataset page <ExternalLink size={13} /></a>
              <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">Dataset license: CC BY 4.0 <ExternalLink size={13} /></a>
            </div>
          </section>
          <section>
            <h3>Technologies &amp; Open-Source Libraries</h3>
            <div className="technology-grid">
              {technologies.map((technology) => (
                <a href={technology.href} target="_blank" rel="noreferrer" key={technology.name} className="technology-card">
                  <span><strong>{technology.name}</strong><small>{technology.role}</small></span>
                  <span>{technology.license} <ExternalLink size={12} /></span>
                </a>
              ))}
            </div>
            <p className="license-note">Dataset and dependency licenses apply independently. Application license: to be determined.</p>
          </section>
          <section className="medical-disclaimer-light">
            <h3>Medical Disclaimer</h3>
            <p>Research prototype only. OsteoInsight is not validated for clinical diagnosis, cancer staging, prognosis, or treatment decisions. Results require professional medical assessment.</p>
          </section>
        </div>
        <footer className="credits-modal-footer">
          <p>Developed by Jonathan Barack Watratan, Rais Akbar Nasution, and Nicholas Azriel Manurung.</p>
          <p>SMA Kemala Taruna Bhayangkara — Bogor, Indonesia.</p>
          <p>Veritas AI Hackathon 2026.</p>
        </footer>
      </div>
    </div>
  )
}
