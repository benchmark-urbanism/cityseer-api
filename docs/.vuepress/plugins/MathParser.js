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


  }
}