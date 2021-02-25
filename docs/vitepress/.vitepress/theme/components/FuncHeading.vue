<template lang="pug">
div
  h5#heading-style
    slot
</template>

<style lang="postcss" scoped>
#heading-style {
  @apply mt-8 text-xl font-semibold border-b-2 border-darkgrey;
}
@media screen and (max-width: 958px) {
  #heading-style {
    @apply text-xl font-semibold;
  }
}
</style>
