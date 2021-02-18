<template lang="pug">
div
  #sig-style
    slot
</template>

<style lang="postcss" scoped>
#sig-style {
  @apply mt-6 mb-3 p-4 text-base font-semibold leading-normal text-white bg-darkgrey rounded-lg;
}

/* necessary for when wrapping signatures with <pre></pre> tags for spacing */
#sig-style pre {
  @apply m-0 p-0;
}
@media screen and (max-width: 950px) {
  #sig-style {
    @apply text-sm;
  }
}
</style>
