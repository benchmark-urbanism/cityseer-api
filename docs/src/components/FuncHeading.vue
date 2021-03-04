<template lang="pug">
div
  h5#heading-style
    slot
</template>

<style lang="postcss" scoped>
#heading-style {
  @apply mt-8 text-xl font-semibold border-b border-darkgrey;
}
@media only screen and (max-width: 958px) {
  #heading-style {
    @apply text-lg font-semibold;
  }
}
</style>
