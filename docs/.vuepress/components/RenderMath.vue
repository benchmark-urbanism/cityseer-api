<template>
    <div></div>
</template>

<script>
  export default {
    name: 'RenderMath',
    mounted () {
      setTimeout(() => {
        window.MathJax.Hub.Queue(['Typeset', MathJax.Hub])
      }, 100)
    }
  }
</script>