import { defineConfig } from 'astro/config'
import { s } from 'hastscript'
import tailwind from '@astrojs/tailwind'
import vue from '@astrojs/vue'

// https://astro.build/config
export default defineConfig({
  // all options are optional; these values are the defaults
  projectRoot: './',
  public: './public',
  dist: './dist',
  src: './src',
  pages: './src/pages',
  integrations: [
    vue(),
    tailwind({
      config: { path: './tailwind.config.js', applyAstroPreset: false },
    }),
  ],
  buildOptions: {
    site: 'https://cityseer.benchmarkurbanism.com/',
    sitemap: true,
    pageUrlFormat: 'directory',
    drafts: false,
  },
  devOptions: {},
  markdownOptions: {
    render: [
      '@astrojs/markdown-remark',
      {
        syntaxHighlight: 'shiki',
        shikiConfig: {
          theme: 'material-darker',
          langs: ['astro'],
          wrap: true,
        },
        remarkPlugins: [
          'remark-gfm',
          'remark-smartypants',
          'remark-emoji',
          '@fec/remark-a11y-emoji',
          [
            'remark-smartypants',
            {
              dashes: 'oldschool',
            },
          ],
          'remark-math',
        ],
        rehypePlugins: [
          'rehype-slug',
          [
            'rehype-katex',
            {
              output: 'html',
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
              csl: 'harvard',
              lang: 'en-GB',
            },
          ],
        ],
      },
    ],
  },
})
