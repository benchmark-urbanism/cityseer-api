let katex = require('katex')

module.exports = {
  base: '/cityseer-api/',
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
  head: [
    // ['link', { rel: 'stylesheet', href: `https://cdnjs.cloudflare.com/...` }],
    // ['script', {src: 'https://cdnjs.cloudflare.com/...'}]
  ],
  dest: 'dist',
  serviceWorker: true,
  markdown: {
    lineNumbers: true,
    anchor: true,
    // https://github.com/cben/mathdown/wiki/math-in-markdown
    extendMarkdown: md => {
      md.use(require('markdown-it-math'), {
        inlineOpen: '$',
        inlineClose: '$',
        blockOpen: '$$',
        blockClose: '$$',
        inlineRenderer: function (str) {
          return katex.renderToString(str, {
            throwOnError: false
          })
        },
        blockRenderer: function (str) {
          return katex.renderToString(str, {
            throwOnError: false,
            displayMode: true
          })
        }
      })
    }
  },
  themeConfig: {
    displayAllHeaders: true,
    lastUpdated: true,
    nav: [
      {text: 'home', link: '/'},
      {text: 'modules', link: '/modules/'},
      {text: 'cityseer.io', link: 'https://cityseer.io/'},
    ],
    sidebarDepth: 1,
    sidebar: [
      '/modules/',
      ['/modules/centrality', '/modules/centrality']
    ]
  },
  evergreen: true
}

/*
markdown-it-math - texzilla renders to mathML - doesn't work with chrome
  - with katex it renders everything doubly - in both MathML and Mathjax?

markdown-it-texmath - causes some sort of odd performance bottleneck
markdown-it-katex - katex less comprehensive but fast - plugin duplicates everything and not maintained
markdown-it-asciimath - not compatible with latex syntax
markdown-it-simplemath - no rendering function
markdown-it-synapse-math - small adaptation of markdown-it-math
*/