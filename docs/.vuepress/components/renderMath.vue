<template>
    <div></div>
</template>

<script>
  export default {
    name: 'renderMath',
    mounted () {
      window.renderMathInElement(document.body)
    }
  }
</script>