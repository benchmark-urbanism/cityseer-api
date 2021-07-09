<template lang="pug">
div
  #arXiv-container
    a#arXiv-link(:href='arXivLink' :title='title' target='_blank')
      img#arXiv-logo(src='../assets/logos/arXiv-logo.png' alt='arXiv logo' width=75)
      #arXiv-text paper identifier: {{ arXivIdentifier }}
</template>

<script>
export default {
  name: 'ArXivLink',
  props: {
    arXivLink: {
      required: true,
      type: String,
    },
  },
  computed: {
    title() {
      return `Link to arXiv pre-print for ${this.arXivLink}`
    },
    arXivIdentifier() {
      const parts = this.arXivLink.split('/')
      return parts[parts.length - 1]
    },
  },
}
</script>

<style lang="postcss">
#arXiv-container {
  @apply flex items-center justify-center my-4;
}
#arXiv-link {
  @apply flex flex-col items-center text-theme relative p-2;
}
#arXiv-logo {
  @apply m-2 object-contain;
}
#arXiv-text {
  @apply text-sm font-medium leading-none;
}
#arXiv-link:hover > #arXiv-text {
  transform: font-bold;
}
</style>
