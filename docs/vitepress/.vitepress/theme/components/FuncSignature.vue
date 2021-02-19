<template lang="pug">
div
  pre
    #sig-style
      slot
</template>

<style lang="postcss" scoped>
#sig-style {
  @apply mt-8 mb-4 py-6 px-4 rounded-lg border border-darkgrey;
  @apply text-base font-semibold leading-normal text-darkgrey bg-lightgrey;
}

/* necessary for when wrapping signatures with <pre></pre> tags for spacing */
#sig-style pre {
  @apply m-0 p-0;
}
@media screen and (max-width: 1024px) {
  #sig-style {
    @apply text-sm;
  }
}
</style>
