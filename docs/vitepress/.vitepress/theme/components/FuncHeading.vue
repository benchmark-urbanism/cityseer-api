<template lang="pug">
div
  #heading-style
    slot
</template>

<style lang="postcss" scoped>
#heading-style {
  @apply py-2 my-2 text-2xl font-semibold border-b-2 border-darkgrey;
}
@media screen and (max-width: 950px) {
  #heading-style {
    @apply text-xl font-semibold;
  }
}
</style>
