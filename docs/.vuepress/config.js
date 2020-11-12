module.exports = {
  base: '/cityseer/',  // must match github pages publish URL
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
  head: [
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css?family=Raleway:300,400,500' }]
  ],
  markdown: {
    lineNumbers: true,
    anchor: true,
    extendMarkdown: md => {}
  },
  plugins: [
    'mathjax', {
      showError: true
    },
    '@vuepress/back-to-top'
  ],
  theme: '@vuepress/theme-default',
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
          ['attribution', 'attribution']
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
    smoothScroll: true,
    serviceWorker: {
      updatePopup: true
    }
  },
  evergreen: true
}
