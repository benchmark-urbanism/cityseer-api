<template lang="pug">
div
  h3#heading-style
    slot
</template>

<style lang="postcss" scoped>
#heading-style {
  @apply pt-4 text-xl font-semibold border-b-2 border-darkgrey;
}
@media screen and (max-width: 958px) {
  #heading-style {
    @apply text-xl font-semibold;
  }
}
</style>
