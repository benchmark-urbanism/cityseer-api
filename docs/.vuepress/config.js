module.exports = {
  base: '/cityseer/',  // must match github pages publish URL
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
  markdown: {
    lineNumbers: true,
    anchor: true,
    extendMarkdown: md => {}
  },
  plugins: ['mathjax', {
    showError: true
  }],
  theme: 'vuepress-theme-cityseer',
  themeConfig: {
    logo: '/round_logo.png',
    lastUpdated: true,
    nav: [
      { text: 'home', link: '/' },
      { text: 'docs', link: '/intro' },
      { text: 'cityseer.io', link: 'https://cityseer.io/' }
    ],
    displayAllHeaders: false,
    activeHeaderLinks: true,
    sidebarDepth: 1,
    sidebar: [
      {
        title: 'overview',
        collapsable: false,
        children: [
          ['intro', 'intro'],
          ['attribution', 'attribution'],
          ['license', 'license']
        ]
      },
      {
        title: 'guide',
        collapsable: false,
        children: [
          ['/guide/cleaning', 'graph cleaning']
        ]
      },
      {
        title: 'cityseer.metrics',
        collapsable: false,
        children: [
          ['/metrics/layers', 'metrics.layers'],
          ['/metrics/networks', 'metrics.networks']
        ]
      },
      {
        title: 'cityseer.util',
        collapsable: false,
        children: [
          ['/util/graphs', 'util.graphs'],
          ['/util/mock', 'util.mock'],
          ['/util/plot', 'util.plot']
        ]
      }
    ],
    repo: 'cityseer/cityseer',
    repoLabel: 'github',
    docsRepo: 'cityseer/cityseer',
    // if your docs are not at the root of the repo:
    docsDir: 'docs',
    // if your docs are in a specific branch (defaults to 'master'):
    docsBranch: 'gh-pages',
    serviceWorker: {
      updatePopup: true
    }
  },
  evergreen: true
}

/*
// auto-render katex clashes with prev / next links


markdown-it-mathjax - parses delimeters and converts to latex standard

  md.use(require('markdown-it-mathjax')())

- need to include CDN link to mathjax script

    ['script', {
      defer: true,
      src: 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_CHTML',
      crossorigin: 'anonymous'
    }]

- doesn't update formulas when changing pages... so need to add MathRendere module

- still doesn't always parse math in modules...!


markdown-it-math - texzilla renders to mathML - doesn't work with chrome
- with katex it renders everything doubly - MUST INCLUDE CSS LINK!!!
- buggy loading - trying web scripts

markdown-it-asciimath - not compatible with latex syntax
markdown-it-simplemath - no rendering function
markdown-it-synapse-math - small adaptation of markdown-it-math

All of the katex based methods seem to cause a major bottleneck with webpack-dev-serve, but build OK
markdown-it-texmath - causes some sort of odd performance bottleneck

md.use(require('markdown-it-texmath').use(require('katex')), {
        delimiters: 'dollars'
      })

    ['link', {
      rel: 'stylesheet',
      href: 'https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/katex.min.css',
      integrity: 'sha384-9eLZqc9ds8eNjO3TmqPeYcDj8n+Qfa4nuSiGYa6DjLNcv9BtN69ZIulL9+8CqC9Y',
      crossorigin: 'anonymous'
    }],

markdown-it-katex - not maintained

@iktakahiro/markdown-it-katex

  md.use(require('@iktakahiro/markdown-it-katex'))

@neilsustc/markdown-it-katex



*/