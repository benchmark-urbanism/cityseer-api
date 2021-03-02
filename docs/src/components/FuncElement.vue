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

<style lang="postcss" scoped>
#container {
  @apply flex py-3 items-start p-1 border-b border-midgrey;
}
#column {
  @apply flex-col py-1;

  min-width: 220px;
  max-width: 300px;
}
#name {
  @apply font-mono text-sm font-semibold;
}
#type {
  @apply font-mono text-xs font-light;
}
#description > p {
  @apply text-sm;
}
#description > * > code,
#description > * > * > code {
  @apply text-xs px-0.5 py-0;
}
@media screen and (max-width: 958px) {
  #container {
    @apply py-1;
  }
  #name {
    @apply text-sm;
  }
}
</style>
