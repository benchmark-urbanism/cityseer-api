<template>
    <div></div>
</template>

<script>
  export default {
    name: 'RenderMath',
    beforeMount () {
      window.renderMathInElement(document.body, {
        delimiters: [
          {left: "$$", right: "$$", display: true}, // has to be first!!
          {left: "$", right: "$", display:false}
        ]
      })
    }
  }
</script>