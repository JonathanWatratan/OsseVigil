import { Bone } from 'lucide-react'

interface LogoProps {
  compact?: boolean
  light?: boolean
}

export function Logo({ compact = false, light = false }: LogoProps) {
  return (
    <div className={`brand-lockup ${compact ? 'brand-lockup--compact' : ''}`} aria-label="OsteoInsight">
      <span className="brand-mark" aria-hidden="true"><Bone size={compact ? 21 : 27} strokeWidth={1.8} /></span>
      {!compact && (
        <span className="brand-copy">
          <strong className={light ? 'text-navy' : ''}>OSTEOINSIGHT</strong>
          <span>Lesion Localization Research</span>
        </span>
      )}
    </div>
  )
}
