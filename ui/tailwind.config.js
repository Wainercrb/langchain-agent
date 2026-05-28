/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#0a0e27',
        'dark-card': '#1a1f3a',
        'dark-border': '#374151',
        'dark-text': '#f3f4f6',
        'dark-text-secondary': '#9ca3af',
      },
    },
  },
  plugins: [],
};
