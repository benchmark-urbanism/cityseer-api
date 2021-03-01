// This is where project configuration and plugin options are located.
// Learn more: https://gridsome.org/docs/config

// Changes here require a server restart.
// To restart press CTRL + C in terminal and run `gridsome develop`
const postcssImport = require('postcss-import')
const tailwindcss = require('tailwindcss')
const autoprefixer = require('autoprefixer')

module.exports = {
  siteName: 'Cityseer API Docs',
  siteDescription:
    'Cityseer API is a collection of computational tools for fine-grained network and land-use analysis, useful for assessing the morphological precursors to vibrant neighbourhoods. It is underpinned by rigorous network-based methods that have been developed from the ground-up specifically for hyperlocal analysis at the pedestrian scale.',
  siteUrl: 'https://cityseer.github.io/',
  pathPrefix: '/cityseer', // only used in production
  plugins: [
    // https://gridsome.org/plugins/@gridsome/source-filesystem
    {
      use: '@gridsome/source-filesystem',
      options: {
        path: '*/*.md',
        typeName: 'Docs',
        baseDir: './content',
        remark: {
          plugins: [],
        },
      },
    },
    {
      use: '@gridsome/source-filesystem',
      options: {
        path: 'landing.md',
        typeName: 'Landing',
        baseDir: './content',
        index: ['landing'],
        remark: {
          plugins: [],
        },
      },
    },
    // https://gridsome.org/plugins/gridsome-plugin-pug
    {
      use: 'gridsome-plugin-pug',
      options: {
        pug: {
          /* Options for `pug-plain-loader` */
        },
        pugLoader: {
          /* Options for `pug-loader` */
        },
      },
    },
    // sitemap
    {
      use: '@gridsome/plugin-sitemap',
      options: {
        cacheTime: 600000, // default
        // exclude: ['/exclude-me'],
        // config: {
        //   '/articles/*': {
        //     changefreq: 'weekly',
        //     priority: 0.5,
        //     lastmod: '2020-02-19',
        //   },
      },
    },
    // robots
    {
      use: 'gridsome-plugin-robots-txt',
      options: {
        policy: [
          {
            userAgent: '*',
            allow: '/',
          },
        ],
      },
    },
  ],
  transformers: {
    remark: {
      plugins: [
        // '@iktakahiro/markdown-it-katex',
        // 'markdown-it-replacements',
        'remark-emoji',
        '@fec/remark-a11y-emoji',
        [
          '@silvenon/remark-smartypants',
          {
            dashes: 'oldschool',
          },
        ],
        [
          'gridsome-plugin-remark-container',
          {
            customTypes: {},
            useDefaultTypes: true, // set to false if you don't want to use default types
            tag: ':::',
            icons: 'none', // can be 'emoji' or 'none'
            classMaster: 'admonition', // generate xyz-content, xyz-icon, xyz-heading
          },
        ],
      ],
    },
  },
  css: {
    loaderOptions: {
      postcss: {
        plugins: [postcssImport, tailwindcss('./tailwind.config.js'), autoprefixer],
      },
    },
  },
}
