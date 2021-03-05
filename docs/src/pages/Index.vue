<template lang="pug">
div
  #home-container
    h1#heading-text {{ tagline }}
    #feature-content
      div(v-for='feature in features')
        h2#feature-text {{ feature.title }}
        p {{ feature.details }}
</template>

<script>
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
}
</script>

<style lang="postcss" scoped>
#home-container {
  @apply flex flex-col items-start pt-72;
}
#heading-text {
  @apply text-4xl font-normal leading-tight;
}
#feature-content {
  @apply flex flex-col max-w-md;
}
#feature-text {
  @apply text-2xl font-medium leading-tight;
}

@media only screen and (max-width: 958px) {
  #home-container {
    @apply pt-8;
  }
  #heading-text {
    @apply text-3xl;
  }
  #feature-content {
    @apply max-w-sm;
  }
  #feature-text {
    @apply text-lg font-semibold;
  }
}
</style>
