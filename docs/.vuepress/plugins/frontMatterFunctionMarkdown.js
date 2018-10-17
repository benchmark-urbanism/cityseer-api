const md = require('markdown-it')({
  html: false,
  linkify: true,
  typographer: true
})

function rend (val) {
  return md.renderInline(val.toString())
}

module.exports = {
  extendPageData ($page) {

    // not all pages have functions
    if ($page.frontmatter.hasOwnProperty('functions')) {

      // iterate function keys
      for (const key of Object.keys($page.frontmatter.functions)) {
        let func = $page.frontmatter.functions[key]

          // render intro text
          func.intro = rend(func.intro)

          // render params
          func.params.forEach(p => {
            for (const key of Object.keys(p)) {
              p[key] = rend(p[key])
            }
          })

          // render returns
          func.returns.forEach(r => {
            for (const key of Object.keys(r)) {
              r[key] = rend(r[key])
            }
          })
        }
    }
  }
}