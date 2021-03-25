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
        xs: '0.75rem',
        sm: '0.85rem',
        base: '0.95rem',
        lg: '1.1rem',
        xl: '1.25rem',
        '2xl': '1.4rem',
        '3xl': '1.6rem',
        '4xl': '2rem',
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
        tight: '1.2',
        snug: '1.4',
        normal: '1.6',
        relaxed: '1.8',
        loose: '2.2',
      },
      borderWidth: {
        default: '1px',
        0: '0',
        1: '1px',
        2: '2px',
        3: '4px',
        4: '6px',
      },
      borderColor: (theme) => ({
        ...theme('colors'),
        DEFAULT: theme('colors.darkgrey', 'currentColor'),
      }),
      colors: {
        theme: '#d32f2f',
        black: '#000',
        lightgrey: '#f5f5f5',
        midgrey: '#dbdbdb',
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
