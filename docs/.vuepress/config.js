module.exports = {
  base: '/cityseer-api/',
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
  head: [
    // must include katex stylesheet otherwise double displayed MathML and HTML
    ['link', {
      rel: 'stylesheet',
      href: 'https://cdn.jsdelivr.net/npm/katex@0.10.0-rc.1/dist/katex.min.css',
      integrity: 'sha384-D+9gmBxUQogRLqvARvNLmA9hS2x//eK1FhVb9PiU86gmcrBrJAQT8okdJ4LMp2uv',
      crossorigin: 'anonymous'
    }],
    ['script', {
      defer: true,
      src: 'https://cdn.jsdelivr.net/npm/katex@0.10.0-rc.1/dist/katex.min.js',
      integrity: 'sha384-483A6DwYfKeDa0Q52fJmxFXkcPCFfnXMoXblOkJ4JcA8zATN6Tm78UNL72AKk+0O',
      crossorigin: 'anonymous'
    }],
    ['script', {
      defer: true,
      src: 'https://cdn.jsdelivr.net/npm/katex@0.10.0-rc.1/dist/contrib/auto-render.min.js',
      integrity: 'sha384-yACMu8JWxKzSp/C1YV86pzGiQ/l1YUfE8oPuahJQxzehAjEt2GiQuy/BIvl9KyeF',
      crossorigin: 'anonymous'
    }]
  ],
  dest: 'dist',
  serviceWorker: true,
  markdown: {
    lineNumbers: true,
    anchor: true,
    // https://github.com/cben/mathdown/wiki/math-in-markdown
    extendMarkdown: md => {
      // mathjax simply converts delimeters to latex compatible form
      // using auto-render katex function from own component to load math in pages
      md.use(require('markdown-it-mathjax')())
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
    ],
    repo: 'cityseer/cityseer-api',
    repoLabel: 'github',
    docsRepo: 'cityseer/cityseer-api',
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
markdown-it-mathjax - simply parses delimeters and converts to latex standard - can then run katex in browser?
markdown-it-math - texzilla renders to mathML - doesn't work with chrome
  - with katex it renders everything doubly - MUST INCLUDE CSS LINK!!!
  - buggy loading - trying web scripts
markdown-it-texmath - causes some sort of odd performance bottleneck
markdown-it-katex - katex less comprehensive but fast - plugin duplicates everything and not maintained
markdown-it-asciimath - not compatible with latex syntax
markdown-it-simplemath - no rendering function
markdown-it-synapse-math - small adaptation of markdown-it-math
*/