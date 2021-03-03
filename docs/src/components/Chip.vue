<template lang="pug">
.inline-block
  #chip-style
    span {{ text }}
</template>

<script>
export default {
  name: 'Chip',
  props: {
    important: {
      type: Boolean,
      required: false,
      default: false,
    },
    text: {
      type: String,
      required: true,
      default: 'txt',
    },
  },
}
</script>

<style lang="postcss" scoped>
#chip-style {
  @apply rounded-sm py-0.5 px-1 text-xs font-bold bg-red-600 text-white inline-block align-top shadow;
}
</style>
