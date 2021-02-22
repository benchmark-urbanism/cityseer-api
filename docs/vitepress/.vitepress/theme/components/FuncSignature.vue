<template lang="pug">
div
  #sig-style
    pre
      slot
</template>

<style lang="postcss" scoped>
#sig-style {
  @apply mt-8 mb-4 p-4 border-t-2 border-darkgrey;
  @apply text-base font-semibold leading-normal text-darkgrey bg-lightgrey;
}

/* necessary for when wrapping signatures with <pre></pre> tags for spacing */
#sig-style pre {
  @apply m-0 p-0;
}
@media screen and (max-width: 958px) {
  #sig-style {
    @apply text-sm;
  }
}
@media screen and (max-width: 600px) {
  #sig-style {
    @apply text-xs;
  }
}
</style>
