<template>
    <div></div>
</template>

<script>
  export default {
    name: 'RenderMath',
    mounted () {
      MathJax.Hub.Queue(['Typeset', MathJax.Hub])
    }
  }
</script>