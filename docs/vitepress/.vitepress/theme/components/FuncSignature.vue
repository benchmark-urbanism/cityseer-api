<template lang="pug">
#sig-style
  slot
</template>

<style lang="postcss" scoped>
#sig-style {
  @apply border-t-3 border-theme text-darkgrey bg-gray-50 mt-1 mb-4 p-4;
  @apply text-sm font-mono font-semibold leading-normal;
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
