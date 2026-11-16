/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        osteo: {
          navy: '#071a2d',
          surface: '#0b2236',
          panel: '#102d46',
          cyan: '#63d3e5',
          muted: '#9fb3c4',
          lesion: '#ff9f43',
        },
      },
    },
  },
  plugins: [],
}
