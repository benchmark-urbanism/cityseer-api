// This is where project configuration and plugin options are located.
// Learn more: https://gridsome.org/docs/config

// Changes here require a server restart.
// To restart press CTRL + C in terminal and run `gridsome develop`
const postcssImport = require('postcss-import')
const tailwindcss = require('@tailwindcss/postcss7-compat')
const autoprefixer = require('autoprefixer')
const remarkBib = require('@benchmark-urbanism/remark-bibtex')

const markdownPlugins = [
  // bibliography
  'remark-footnotes',
  [
    remarkBib,
    {
      bibtexFile: 'src/assets/bib/mendeley.bib',
    },
  ],
  // autolink headings - included in transformer remark by default but want clickable link symbol
  [
    'remark-autolink-headings',
    {
      content: {
        type: 'element',
        tagName: 'font-awesome',
        properties: { icon: 'link', size: 'xs', class: ['text-theme', 'px-1', 'mx-1'] },
      },
    },
  ],
  // emojis
  'remark-emoji',
  // makes emojis accessible
  '@fec/remark-a11y-emoji',
  // em-dashes and ellipses, etc.
  [
    '@silvenon/remark-smartypants',
    {
      dashes: 'oldschool',
    },
  ],
  // note boxes
  [
    'gridsome-plugin-remark-container',
    {
      customTypes: {},
      useDefaultTypes: true, // set to false if you don't want to use defaults
      tag: ':::',
      icons: 'none', // can be 'emoji' or 'none'
      classMaster: 'admonition', // generate xyz-content, xyz-icon, heading, etc.
    },
  ],
  // code blogs
  '@gridsome/remark-prismjs',
  // latex maths
  'gridsome-remark-katex',
]

module.exports = {
  siteName: 'Cityseer API Docs',
  siteDescription:
    'Cityseer API is a collection of computational tools for fine-grained network and land-use analysis, useful for assessing the morphological precursors to vibrant neighbourhoods. It is underpinned by rigorous network-based methods that have been developed from the ground-up specifically for hyperlocal analysis at the pedestrian scale.',
  siteUrl: 'https://cityseer.benchmarkurbanism.com/',
  plugins: [
    {
      use: '@gridsome/vue-remark',
      options: {
        typeName: 'Doc',
        baseDir: './content',
        template: './src/templates/Doc.vue',
        plugins: markdownPlugins,
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
  permalinks: {
    trailingSlash: true,
  },
  css: {
    loaderOptions: {
      postcss: {
        plugins: [postcssImport, tailwindcss('./tailwind.config.js'), autoprefixer],
      },
    },
  },
  icon: './src/favicon.png',
}
