<template lang="pug">
div
  main#layout-container
    slot
    section
      slot(name='left')
</template>

<script>
export default {
  name: 'DefaultLayout',
}
</script>

<style lang="postcss" scoped></style>
