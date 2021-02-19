<template lang="pug">
div
  #sig-style
    slot
</template>

<style lang="postcss" scoped>
#sig-style {
  @apply mt-6 mb-3 p-4 text-base font-semibold leading-normal text-white bg-darkgrey rounded-lg;
}

@media screen and (max-width: 1024px) {
  #sig-style {
    @apply text-sm;
  }
}
@media screen and (max-width: 900px) {
  #sig-style {
    @apply text-xs;
  }
}
</style>
