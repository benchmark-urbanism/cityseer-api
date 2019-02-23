module.exports = {
  base: '/cityseer/',  // must match github pages publish URL
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
  head: [
    ['script', {
      defer: true,
      src: 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_CHTML',
      crossorigin: 'anonymous'
    }]
  ],
  plugins: [],
  markdown: {
    lineNumbers: true,
    anchor: true,
    extendMarkdown: md => {
      md.use(require('markdown-it-mathjax')())
    }
  },
  theme: 'vuepress-theme-cityseer',
  themeConfig: {
    logo: '/round_logo.png',
    lastUpdated: true,
    nav: [
      {text: 'home', link: '/'},
      {text: 'docs', link: '/intro'},
      {text: 'cityseer.io', link: 'https://cityseer.io/'}
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
// using auto-render katex function from own component to load math in pages
// created plugin for parsing front matter data for functions

// https://github.com/cben/mathdown/wiki/math-in-markdown
// mathjax simply converts delimeters to latex compatible form
// md.use(require('markdown-it-mathjax')())

markdown-it-mathjax - simply parses delimeters and converts to latex standard - can then run katex in browser?
markdown-it-math - texzilla renders to mathML - doesn't work with chrome
  - with katex it renders everything doubly - MUST INCLUDE CSS LINK!!!
  - buggy loading - trying web scripts

All of the katex based methods seem to cause a major bottleneck with webpack-dev-serve, but build OK
markdown-it-texmath - causes some sort of odd performance bottleneck

md.use(require('markdown-it-texmath').use(require('katex')), {
        delimiters: 'dollars'
      })

markdown-it-katex - katex less comprehensive but fast - plugin duplicates everything and not maintained
markdown-it-asciimath - not compatible with latex syntax
markdown-it-simplemath - no rendering function
markdown-it-synapse-math - small adaptation of markdown-it-math
@iktakahiro/markdown-it-katex


*/