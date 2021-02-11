// https://github.com/vuejs/vitepress/blob/master/src/node/config.ts
module.exports = {
    lang: 'en-GB',
    base: '/cityseer/',  // must match github pages publish URL
    title: 'Cityseer API Docs',
    description: 'Computational tools for urban analysis',
    head: [
        ['link', {rel: 'stylesheet', href: 'https://fonts.googleapis.com/css?family=Raleway:300,400,500'}],
        ['link', {rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.11.1/katex.min.css'}],
        ['link', {rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/4.0.0/github-markdown.min.css'}]
    ],
    themeConfig: {
        locales: {},
        repo: 'cityseer/cityseer',
        docsDir: 'content',
        algolia: {
            apiKey: 'f7782fe88b05e9d75005c2b23cdb27e5',
            indexName: 'cityseer_docs'
        },
        searchPlaceholder: 'search',
        searchMaxSuggestions: 10,
        logo: '/round_logo.png',
        lastUpdated: true,
        nav: [
            {text: 'home', link: '/'},
            {text: 'docs', link: '/intro'},
            {text: 'cityseer.io', link: 'https://cityseer.io/'}
        ],
        displayAllHeaders: false,
        activeHeaderLinks: true,
        sidebar: [
            {
                text: 'overview',
                children: [
                    {text: 'intro', link: 'intro'},
                    {text: 'attribution', link: 'attribution'}
                ]
            },
            {
                text: 'guide',
                children: [
                    {text: 'graph cleaning', link: '/guide/cleaning'}
                ]
            },
            {
                text: 'cityseer.metrics',
                children: [
                    {text: 'metrics.layers', link: '/metrics/layers'},
                    {text: 'metrics.networks', link: '/metrics/networks'}
                ]
            },
            {
                text: 'cityseer.util',
                children: [
                    {text: 'util.graphs', link: '/util/graphs'},
                    {text: 'util.mock', link: '/util/mock'},
                    {text: 'util.plot', link: '/util/plot'}
                ]
            }
        ],
    },
    locales: {},
    markdown: {
        lineNumbers: true,
        toc: {includeLevel: [1, 2]},
        config: (md) => {
            md.use(require('@iktakahiro/markdown-it-katex'))
        },
    },
}
