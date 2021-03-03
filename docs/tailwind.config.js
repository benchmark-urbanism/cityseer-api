module.exports = {
  purge: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  important: true,
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
        base: '1.0rem',
        lg: '1.1rem',
        xl: '1.3rem',
        '2xl': '1.6rem',
        '3xl': '1.8rem',
        '4xl': '2.4rem',
      },
      fontWeight: {
        extralight: '200',
        light: '300',
        normal: '400',
        medium: '500',
        semibold: '600',
        bold: '700',
      },
      // tracking
      letterSpacing: {
        tighter: '-0.05em',
        tight: '-0.025em',
        normal: '0',
        wide: '0.025em',
        wider: '0.05em',
      },
      // leading
      lineHeight: {
        none: '1',
        tight: '1.25',
        snug: '1.5',
        normal: '1.75',
        relaxed: '2',
        loose: '2.25',
      },
      borderWidth: {
        default: '1px',
        0: '0',
        1: '1px',
        2: '2px',
        3: '4px',
        4: '6px',
      },
      colors: {
        theme: '#d32f2f',
        black: '#000',
        lightgrey: '#f5f5f5',
        midgrey: '#ebebeb',
        darkgrey: '#2e2e2e',
        white: '#fff',
      },
    },
  },
  variants: {
    extend: {},
  },
  plugins: [],
}
