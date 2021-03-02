<template lang="pug">
Layout
  template
    #landing-content
      div
        #tagline
          h1 {{ landing.tagline }}
        #features
          div(v-for='feature in landing.features')
            h2.feature-heading {{ feature.title }}
            p.feature-blurb {{ feature.details }}
</template>

<page-query>
query ($id: ID) {
  landing: landing (id: $id) {
    id
    tagline
    features {
      title
      details
    }
  }
  docs: allDoc {
    edges {
      node {
        id
        path
        fileInfo {
          name
          path
        }
        title
      }
    }
  }
}
</page-query>

<script>
import { mapGetters } from 'vuex'

export default {
  name: 'Landing',
  computed: {
    ...mapGetters(['smallMode', 'makeMonthDate']),
    landing() {
      return this.$page.landing
    },
    docs() {
      return this.$page.docs.edges.map((edge) => edge.node)
    },
  },
}
</script>

<style lang="postcss" scoped>
#landing-content {
  @apply w-full flex justify-center bg-red-100;
}
#tagline {
  @apply text-2xl text-right font-semibold px-8 py-6;
}
#features {
  @apply flex;
}
.feature-heading {
  @apply text-xl font-semibold;
}
.feature-blurb {
}

@media only screen and (max-width: 1500px) {
  #page-title {
    @apply text-4xl font-light px-2 pt-4;
  }
  #intro-text {
    @apply text-xl px-6 py-2 font-medium;
  }
  .row {
    @apply py-6;
  }
  .trip-title,
  .excerpt {
    @apply px-6;
  }
  .trip-title {
    @apply text-xl;
  }
}

@media only screen and (max-height: 958px) {
  #page-title {
    @apply text-4xl font-light px-2 pt-4;
  }
  #intro-text {
    @apply text-right text-lg font-medium px-6 py-2;
  }
  .row {
    @apply py-6;
  }
  .trip-title,
  .excerpt {
    @apply text-base px-6;
  }
  .trip-title {
    @apply text-xl;
  }
}

@media only screen and (max-width: 958px) {
  #page-title {
    @apply text-xl py-4 font-medium;
  }
  #intro-text {
    @apply text-left text-sm font-medium p-2 pl-3;
  }
  .row {
    @apply py-6;
  }
  .trip-title,
  .excerpt {
    @apply text-sm px-4;
  }
  .trip-title {
    @apply text-xl;
  }
}
</style>
