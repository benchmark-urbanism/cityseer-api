<template lang="pug">
div
  #home-container
    h1.text-4xl.font-normal.leading-tight.pt-32.pb-16 {{ tagline }}
    #feature-content
      div(v-for='feature in features')
        h2.text-2xl.font-medium.leading-tight {{ feature.title }}
        p {{ feature.details }}
</template>

<script>
import { mapGetters } from 'vuex'

export default {
  name: 'Home',
  data() {
    return {
      tagline: 'Computational tools for urban analysis',
      features: [
        {
          title: 'Urban morphological analysis',
          details:
            'Tools for fine-grained network and land-use analysis. Assess the morphological precursors to vibrant neighbourhoods.',
        },
        {
          title: 'Pedestrian scale',
          details:
            'Rigorous network-based methods developed specifically from the ground-up for high-resolution localised analysis at the pedestrian scale.',
        },
        {
          title: 'Friendly & performant',
          details:
            'Python package with heavier algorithms optimised in fast JIT compiled code. In-out convenience methods for NetworkX.',
        },
      ],
    }
  },
  computed: {
    ...mapGetters(['smallMode', 'makeMonthDate']),
    landing() {
      console.log(this.$page.landing)
      return this.$page.landing
    },
    docs() {
      return this.$page.docs.edges.map((edge) => edge.node)
    },
  },
}
</script>

<style lang="postcss" scoped>
#home-container {
  @apply flex flex-col items-start;
}
#feature-content {
  @apply flex flex-col max-w-md;
}

@media only screen and (max-width: 958px) {
  #home-container {
    @apply pt-8;
  }
  #feature-content {
    @apply max-w-sm;
  }
}
</style>
