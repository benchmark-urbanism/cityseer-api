<template>
    <div></div>
</template>

<script>
  export default {
    name: 'RenderMath',
    mounted () {
      window.MathJax.Hub.Queue(['Typeset', MathJax.Hub])
    }
  }
</script>