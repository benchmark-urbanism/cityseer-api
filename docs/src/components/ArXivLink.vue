<template lang="pug">
div.flex.justify-center.py-4
  a#arXiv-container(
    :href='arXivLink',
    :title='title'
    target='_blank'
  )
    span#arXiv-logo arXiv.org
    div.text-sm.font-medium.leading-none arXiv id: {{ arXivIdentifier }}
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  arXivLink: {
    required: true,
    type: String,
  },
})
const title = computed(() => `Link to arXiv pre-print for ${props.arXivLink}`)
const arXivIdentifier = computed(() => {
  const parts = props.arXivLink.split('/')
  return parts[parts.length - 1]
})
</script>

<style lang="postcss" scoped>
#arXiv-container {
  @apply flex flex-col items-center justify-center;
  @apply m-2 rounded-sm border-0.5 border-mid-grey bg-dark-grey p-2 shadow;
}
#arXiv-logo {
  @apply m-2 bg-theme px-1 font-mono text-white transition-all;
}
#arXiv-container:hover > #arXiv-logo {
  @apply scale-105;
}
</style>
