export interface TechnologyCredit {
  name: string
  role: string
  license: string
  href: string
}

export const technologies: TechnologyCredit[] = [
  { name: 'Python', role: 'Backend and ML runtime', license: 'PSF License', href: 'https://docs.python.org/3/license.html' },
  { name: 'PyTorch', role: 'Machine-learning runtime', license: 'BSD-3-Clause', href: 'https://github.com/pytorch/pytorch/blob/main/LICENSE' },
  { name: 'Ultralytics YOLO', role: 'Archived localization research', license: 'AGPL-3.0', href: 'https://github.com/ultralytics/ultralytics/blob/main/LICENSE' },
  { name: 'FastAPI', role: 'Classification API', license: 'MIT', href: 'https://github.com/fastapi/fastapi/blob/master/LICENSE' },
  { name: 'OpenCV', role: 'Image evaluation utilities', license: 'Apache-2.0', href: 'https://github.com/opencv/opencv/blob/4.x/LICENSE' },
  { name: 'React', role: 'Application interface', license: 'MIT', href: 'https://github.com/facebook/react/blob/main/LICENSE' },
  { name: 'TypeScript', role: 'Typed frontend code', license: 'Apache-2.0', href: 'https://github.com/microsoft/TypeScript/blob/main/LICENSE.txt' },
  { name: 'Vite', role: 'Frontend tooling', license: 'MIT', href: 'https://github.com/vitejs/vite/blob/main/LICENSE' },
  { name: 'Tailwind CSS', role: 'Design utilities', license: 'MIT', href: 'https://github.com/tailwindlabs/tailwindcss/blob/main/LICENSE' },
  { name: 'Lucide', role: 'Interface icons', license: 'ISC', href: 'https://github.com/lucide-icons/lucide/blob/main/LICENSE' },
  { name: 'Motion for React', role: 'Accessible interface motion', license: 'MIT', href: 'https://github.com/motiondivision/motion/blob/main/LICENSE.md' },
]
