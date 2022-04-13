module.exports = {
  content: ['./public/**/*.html', './src/**/*.{astro,html,md,js,jsx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      fontFamily: {
        defaults: ['Raleway', 'serif'],
        headings: ['Raleway', 'sans-serif'],
        mono: ['Roboto Mono', 'monospace'],
      },
      fontSize: {
        xxs: '0.7rem',
        xs: '0.8rem',
        sm: '0.9rem',
        base: '1rem',
        lg: '1.05rem',
        xl: '1.1rem',
        '2xl': '1.1rem',
        '3xl': '1.3rem',
        '4xl': '2.2rem',
      },
      fontWeight: {
        extralight: '200',
        light: '300',
        normal: '400',
        medium: '500',
        semibold: '600',
        bold: '700',
      },
      borderWidth: {
        default: '1px',
        0: '0',
        0.5: '0.5px',
        1: '1px',
        2: '2px',
        3: '3px',
        4: '4px',
      },
      borderColor: (theme) => ({
        ...theme('colors'),
        DEFAULT: theme('colors.darker-grey', 'currentColor'),
      }),
      colors: {
        theme: '#D32333',
        'lighter-grey': '#F1F1F1',
        'light-grey': '#A8A8A8',
        'mid-grey': '#404040',
        'dark-grey': '#1D1D1D',
        'darker-grey': '#19181B',
      },
    },
  },
}
