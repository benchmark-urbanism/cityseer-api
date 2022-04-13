import sitemap from '@astrojs/sitemap'
import tailwind from '@astrojs/tailwind'
import vue from '@astrojs/vue'
import { defineConfig } from 'astro/config'
import { h, s } from 'hastscript'
import { visit } from 'unist-util-visit'

function admonitionRemarkPlugin() {
  return (tree) => {
    visit(tree, (node) => {
      if (
        node.type === 'textDirective' ||
        node.type === 'leafDirective' ||
        node.type === 'containerDirective'
      ) {
        if (node.name === 'note') {
          const data = node.data || (node.data = {})
          const tagName = node.type === 'textDirective' ? 'span' : 'div'
          data.hName = tagName
          data.hProperties = h(tagName, { class: 'box note' }).properties
        } else if (node.name === 'warning') {
          const data = node.data || (node.data = {})
          const tagName = node.type === 'textDirective' ? 'span' : 'div'
          data.hName = tagName
          data.hProperties = h(tagName, { class: 'box warning' }).properties
        } else {
          return
        }
      }
    })
  }
}

export default defineConfig({
  root: '.',
  srcDir: './src',
  publicDir: './public',
  outDir: './dist',
  site: 'https://cityseer.benchmarkurbanism.com/',
  base: '/',
  trailingSlash: 'always',
  build: {
    format: 'directory',
  },
  markdown: {
    drafts: false,
    syntaxHighlight: 'shiki',
    shikiConfig: {
      theme: 'material-darker',
      langs: ['astro'],
      wrap: true,
    },
    remarkPlugins: [
      'remark-gfm',
      'remark-emoji',
      '@fec/remark-a11y-emoji',
      [
        'remark-smartypants',
        {
          dashes: 'oldschool',
        },
      ],
      'remark-math',
      'remark-directive',
      admonitionRemarkPlugin,
    ],
    rehypePlugins: [
      'rehype-slug',
      [
        'rehype-katex',
        {
          output: 'htmlAndMathml',
        },
      ],
      [
        'rehype-autolink-headings',
        {
          test: ['h1', 'h2', 'h3'],
          behavior: 'prepend',

          content(node) {
            return [
              s(
                'svg',
                {
                  xmlns: 'http://www.w3.org/2000/svg',
                  viewbox: '0 0 20 20',
                  ariaHidden: 'true',
                  width: '15px',
                  height: '15px',
                  class: 'heading-icon',
                },
                [
                  s('title', 'SVG `<path>` element'),
                  s('path', {
                    d: 'M12.586 4.586a2 2 0 112.828 2.828l-3 3a2 2 0 01-2.828 0 1 1 0 00-1.414 1.414 4 4 0 005.656 0l3-3a4 4 0 00-5.656-5.656l-1.5 1.5a1 1 0 101.414 1.414l1.5-1.5zm-5 5a2 2 0 012.828 0 1 1 0 101.414-1.414 4 4 0 00-5.656 0l-3 3a4 4 0 105.656 5.656l1.5-1.5a1 1 0 10-1.414-1.414l-1.5 1.5a2 2 0 11-2.828-2.828l3-3z',
                    'fill-rule': 'evenodd',
                    'clip-rule': 'evenodd',
                  }),
                ]
              ),
            ]
          },
        },
      ],
      [
        'rehype-citation',
        {
          bibliography: './src/assets/bib/mendeley.bib',
          csl: 'harvard1',
          lang: 'en-US',
        },
      ],
    ],
  },
  integrations: [
    vue(),
    tailwind({
      config: {
        path: './tailwind.config.cjs',
        applyAstroPreset: false,
        applyBaseStyles: false,
      },
    }),
    sitemap(),
  ],
})
