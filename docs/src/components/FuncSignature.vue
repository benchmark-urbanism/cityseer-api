<template lang="pug">
div
  .w-full.border-b-3.border-theme.h-0
  #sig-style
    slot
</template>

<style lang="postcss" scoped>
#sig-style {
  @apply mb-4 py-3 px-4 overflow-x-auto whitespace-pre-wrap;
  @apply border border-midgrey bg-gray-200 rounded-br-md rounded-bl-md;
  @apply text-sm font-mono font-semibold leading-normal;
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
