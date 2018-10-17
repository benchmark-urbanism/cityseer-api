<template>
    <div></div>
</template>

<script>
  export default {
    name: 'RenderMath',
    mounted () {
      window.renderMathInElement(document.body, {
        delimiters: [
          {left: "$$", right: "$$", display: true}, // has to be first!!
          {left: "$", right: "$", display:false}
        ]
      })
    }
  }
</script>