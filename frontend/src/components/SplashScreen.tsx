import { motion, useReducedMotion } from 'framer-motion'
import { Logo } from './Logo'

export function SplashScreen() {
  const reduceMotion = useReducedMotion()

  return (
    <motion.div
      className="splash-screen"
      initial={{ opacity: 1 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: reduceMotion ? 0.01 : 0.3 }}
      aria-label="OsteoInsight loading"
    >
      <motion.div
        className="splash-content"
        initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: reduceMotion ? 0.01 : 0.3 }}
      >
        <Logo />
        <p>AI-Assisted Bone Tumor Analysis</p>
        <div className="pulse-line" aria-hidden="true">
          <svg viewBox="0 0 320 48" preserveAspectRatio="none">
            <motion.path
              d="M0 25 H72 L88 25 L98 9 L112 39 L126 20 L138 25 H320"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              initial={{ pathLength: reduceMotion ? 1 : 0, opacity: 0.45 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ delay: reduceMotion ? 0 : 0.3, duration: reduceMotion ? 0.01 : 0.7 }}
            />
          </svg>
        </div>
      </motion.div>
    </motion.div>
  )
}
