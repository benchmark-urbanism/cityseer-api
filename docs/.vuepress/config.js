module.exports = {
  base: '/cityseer/',  // must match github pages publish URL
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
  head: [
    ['link', {
      rel: 'stylesheet',
      href: 'https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/katex.min.css',
      integrity: 'sha384-dbVIfZGuN1Yq7/1Ocstc1lUEm+AT+/rCkibIcC/OmWo5f0EA48Vf8CytHzGrSwbQ',
      crossorigin: 'anonymous'
    }]
  ],
  plugins: [],
  markdown: {
    lineNumbers: true,
    anchor: true,
    extendMarkdown: md => {
      md.use(require('markdown-it-texmath').use(require('katex')), {
        delimiters: 'dollars'
      })
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
