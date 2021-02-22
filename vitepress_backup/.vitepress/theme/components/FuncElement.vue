<template lang="pug">
div
  #container
    #column
      #name
        span {{ name }}
      #type
        span {{ type }}
    #description
      slot
</template>

<style lang="postcss" scoped>
#container {
  @apply flex py-3 items-start p-1 border-b border-lightgrey;
}
#column {
  @apply flex-col py-1;

  min-width: 220px;
  max-width: 300px;
}
#name {
  @apply font-mono text-base font-bold;
}
#type {
  @apply font-mono text-xs font-normal;
}
#description {
  @apply text-sm;
}
@media screen and (max-width: 1024px) {
  #container {
    @apply py-1;
  }
  #name {
    @apply text-sm;
  }
}
</style>

<script>
export default {
  name: 'DisplayFunction',
  props: {
    name: {
      type: String,
      required: true,
    },
    type: {
      type: String,
      required: false,
      default: '',
    },
  },
}
</script>
