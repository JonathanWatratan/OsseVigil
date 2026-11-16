import { AlertTriangle, CheckCircle2, Database, Microscope, Network } from 'lucide-react'
import type { HealthResponse } from '../types'

export function ResearchPage({ health }: { health: HealthResponse | null }) {
  const ready = health?.status === 'ready'
  return (
    <div className="content-page">
      <section className="page-intro"><span className="eyebrow">SCIENTIFIC TRANSPARENCY</span><h1>Research Information</h1><p>Verified project state, intended evaluation, data provenance, and limitations.</p></section>
      <div className="research-grid">
        <article className="info-card feature-card"><Microscope size={23} /><span className="card-kicker">ACTIVE PRODUCT TASK</span><h2>Dataset-category classification</h2><p>The active ResNet18 compares a radiograph with the Roboflow dataset's Cancer and Normal categories. These labels and scores are not independently verified diagnoses or clinical probabilities.</p><span className={`research-state ${ready ? 'ready' : ''}`}>{ready ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}{ready ? `Classifier loaded · ${health?.model_version}` : 'Classifier checkpoint not available'}</span></article>
        <article className="info-card"><Database size={22} /><span className="card-kicker">ACTIVE DATASET</span><h2>Roboflow experiment</h2><p>The remediated working set retains 5,823 representatives after duplicate handling. Patient independence and clinical annotation provenance could not be established.</p></article>
        <article className="info-card"><Network size={22} /><span className="card-kicker">HELD-OUT EVALUATION</span><h2>Exploratory metrics</h2><p>On 593 source-grouped test images, macro-F1 was 0.874 and balanced accuracy was 0.874. Source and anatomy confounding mean these are not diagnostic-accuracy estimates.</p></article>
      </div>
      <section className="limitations-panel"><h2>Current limitations</h2><ul><li>No independent fracture/non-cancer-abnormality challenge set has been evaluated.</li><li>No reliable patient identifiers are available, so patient-independent performance is unknown.</li><li>Source family and anatomical region remain correlated with dataset labels.</li><li>Scores are uncalibrated and may fail on unfamiliar hospitals, equipment, anatomy, fractures, or infections.</li><li>BTXRD localization research files remain archived, but localization is not part of this MVP.</li></ul></section>
    </div>
  )
}
