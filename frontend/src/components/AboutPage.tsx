import { Bone, HeartPulse, ShieldCheck } from 'lucide-react'

export function AboutPage() {
  return (
    <div className="content-page">
      <section className="page-intro"><span className="eyebrow">ABOUT THE PROJECT</span><h1>About OsteoInsight</h1><p>A focused medical-imaging research prototype for the Veritas AI International High School Hackathon 2026.</p></section>
      <section className="about-hero"><div className="about-mark"><Bone size={44} /></div><div><h2>A focused research classifier</h2><p>OsteoInsight explores a lightweight model that compares bone radiographs with two dataset-provided categories while making its uncertainty and evidence limits visible.</p></div></section>
      <div className="about-grid">
        <article><HeartPulse size={22} /><h3>Human review remains essential</h3><p>Category scores cannot replace radiologists, pathology, clinical history, or additional imaging.</p></article>
        <article><ShieldCheck size={22} /><h3>Safety by design</h3><p>The app labels scores as research outputs and provides no cancer confirmation, stage, treatment, prognosis, urgency, or chemotherapy-eligibility output.</p></article>
      </div>
      <section className="project-team"><span className="card-kicker">PROJECT DEVELOPERS</span><h2>Jonathan Barack Watratan · Rais Akbar Nasution · Nicholas Azriel Manurung</h2><p>SMA Kemala Taruna Bhayangkara — Gunung Sindur, Kabupaten Bogor, Jawa Barat, Indonesia.</p></section>
    </div>
  )
}
