module.exports = {
  base: '/cityseer-api/',
  title: 'Cityseer API Docs',
  description: 'Computational tools for urban analysis',
  markdown: {
    anchor: true,
    lineNumbers: true,
    extendMarkdown: md => {
      md.use(require('markdown-it-katex'), { 'throwOnError' : false, 'errorColor' : '#cc0000' })
    }
  },
  evergreen: true
}