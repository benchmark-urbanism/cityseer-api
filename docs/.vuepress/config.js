const VuetifyLoaderPlugin = require('vuetify-loader/lib/plugin')

module.exports = {
  base: '/cityseer/',  // must match github pages publish URL
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
  head: [
    // must include katex stylesheet otherwise double displayed MathML and HTML
    ['link', {
      rel: 'stylesheet',
      href: 'https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/katex.min.css',
      integrity: 'sha384-9eLZqc9ds8eNjO3TmqPeYcDj8n+Qfa4nuSiGYa6DjLNcv9BtN69ZIulL9+8CqC9Y',
      crossorigin: 'anonymous'
    }],
    ['script', {
      defer: true,
      src: 'https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/katex.min.js',
      integrity: 'sha384-K3vbOmF2BtaVai+Qk37uypf7VrgBubhQreNQe9aGsz9lB63dIFiQVlJbr92dw2Lx',
      crossorigin: 'anonymous'
    }],
    ['script', {
      defer: true,
      src: 'https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/contrib/auto-render.min.js',
      integrity: 'sha384-kmZOZB5ObwgQnS/DuDg6TScgOiWWBiVt0plIRkZCmE6rDZGrEOQeHM5PcHi+nyqe',
      crossorigin: 'anonymous'
    }]
  ],
  plugins: [
    // require('./plugins/MathParser.js')
  ],
  serviceWorker: true,
  markdown: {
    lineNumbers: true,
    anchor: true,
    extendMarkdown: md => {}
  },
  extend: '@vuepress/theme-default',
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
          ['intro', 'intro']
        ]
      },
      {
        title: 'algos',
        collapsable: false,
        children: [
          ['/algos/centrality', 'algos.centrality'],
          ['/algos/checks', 'algos.checks'],
          ['/algos/data', 'algos.data'],
          ['/algos/diversity', 'algos.diversity']
        ]
      },
      {
        title: 'metrics',
        collapsable: false,
        children: [
          ['/metrics/layers', 'metrics.layers'],
          ['/metrics/networks', 'metrics.networks']
        ]
      },
      {
        title: 'util',
        collapsable: false,
        children: [
          ['/util/graphs', 'util.graphs'],
          ['/util/mock', 'util.mock'],
          ['/util/plot', 'util.plot']
        ]
      }
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
  //evergreen: true,
  configureWebpack: (config, isServer) => {
    if (!isServer) {
      // mutate the config for client
      config.plugins.push(new VuetifyLoaderPlugin())
    }
  }
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
markdown-it-texmath - causes some sort of odd performance bottleneck
markdown-it-katex - katex less comprehensive but fast - plugin duplicates everything and not maintained
markdown-it-asciimath - not compatible with latex syntax
markdown-it-simplemath - no rendering function
markdown-it-synapse-math - small adaptation of markdown-it-math
*/