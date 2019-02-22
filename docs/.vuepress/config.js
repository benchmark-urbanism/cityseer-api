module.exports = {
  base: '/cityseer/',  // must match github pages publish URL
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
  head: [
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
    // must include katex stylesheet otherwise double displayed MathML and HTML
    ['script', {
      defer: true,
      src: 'https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/contrib/auto-render.min.js',
      integrity: 'sha384-kmZOZB5ObwgQnS/DuDg6TScgOiWWBiVt0plIRkZCmE6rDZGrEOQeHM5PcHi+nyqe',
      crossorigin: 'anonymous'
    }]
  ],
  plugins: [],
  markdown: {
    lineNumbers: true,
    anchor: true,
    extendMarkdown: md => {}
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
