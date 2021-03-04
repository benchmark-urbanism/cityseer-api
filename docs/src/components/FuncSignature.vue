<template lang="pug">
div
  .w-full.border-b-3.border-theme.h-0
  #sig-style
    slot
</template>

<style lang="postcss" scoped>
#sig-style {
  @apply border-t border-lightgrey bg-midgrey mt-0.5 mb-4 px-4 py-2;
  @apply text-sm font-mono font-semibold leading-normal overflow-auto;
}

/* necessary for when wrapping signatures with <pre></pre> tags for spacing */
#sig-style pre {
  @apply m-0 p-0;
}
@media screen and (max-width: 958px) {
  #sig-style {
    @apply text-xs;
  }
}
</style>
