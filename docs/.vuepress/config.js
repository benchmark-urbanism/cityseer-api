const VuetifyLoaderPlugin = require('vuetify-loader/lib/plugin')

module.exports = {
  base: '/cityseer/',  // must match github pages publish URL
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
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
  evergreen: true,
  configureWebpack: (config, isServer) => {
    // Vuepress is SSR - isServer refers to whether server or client side
    if (!isServer) {
      // mutate the config for client
      config.plugins.push(new VuetifyLoaderPlugin())
    }
  }
}
