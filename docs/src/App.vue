<template lang="pug">
// don't use id# because overriden (#app) from elsewhere
.app-container
  // container for logo
  header.flex
    router-link#logo-route(
      to='/'
      :class='{ "pointer-events-none": $route.path === "/" }'
      title='home'
    )
      img#logo-img.foreground-pulse(src='./assets/logos/cityseer_logo_deep_red.png' alt='logo')
  main.flex-1.flex
    router-view#app-content
  footer#footer-container
    div Copyright © 2018-present Gareth Simons
</template>

<static-query>
query {
  metadata {
    siteName
    siteDescription
    siteUrl
    pathPrefix
  }
}
</static-query>

<script>
import { mapGetters } from 'vuex'
import { throttle } from 'lodash'
import anime from 'animejs/lib/anime.es.js'

export default {
  name: 'App',
  metaInfo() {
    return {
      title: 'Hello!',
      titleTemplate: `%s | ${this.$static.metadata.siteName}`,
      meta: [
        // use the key for overriding from child components
        {
          key: 'description',
          name: 'description',
          content: this.$static.metadata.siteDescription,
        },
        {
          name: 'keywords',
          content:
            'cityseer, urban, metrics, analytics, big data, predictive analytics, urban design, planning, property development, machine learning',
        },
      ],
    }
  },
  data() {
    return {
      logoLarge: true,
      largeSize: 500,
      smallSize: 200,
    }
  },
  computed: {
    ...mapGetters(['smallMode']),
  },
  watch: {
    smallMode() {
      if (!process.isClient) return
      this.updateLogoSize()
    },
  },
  mounted() {
    // check that logo is set to small if loading from non-home route
    if (!process.isClient) return
    // keep store's references to dom refreshed
    // used for updating page layouts for small screens
    this.domDims()
    window.addEventListener('resize', () => this.domDims())
    // check the logo size is in sync
    this.updateLogoSize()
  },
  unmounted() {
    if (!process.isClient) return
    window.removeEventListener('resize', () => this.domDims())
  },
  methods: {
    domDims: throttle(function () {
      this.$store.commit('setDomDims', {
        width: window.innerWidth,
        height: window.innerHeight,
      })
    }, 50),
    updateLogoSize() {
      if (!this.smallMode && !this.logoLarge) {
        this.toLargeLogo()
      } else if (this.smallMode && this.logoLarge) {
        this.toSmallLogo()
      }
    },
    toSmallLogo() {
      anime({
        targets: '#logo-route',
        width: this.smallSize,
        duration: 500,
        complete: () => {
          this.logoLarge = false
        },
      })
    },
    toLargeLogo() {
      anime({
        targets: '#logo-route',
        width: this.largeSize,
        duration: 500,
        complete: () => {
          this.logoLarge = true
        },
      })
    },
  },
}
</script>

<style lang="postcss" scoped>
.app-container {
  @apply min-h-screen flex flex-col;
}

#logo-route {
  @apply relative flex items-end m-4;

  width: 500px;
}

#logo-img {
  @apply z-10 transition-all w-full object-contain;
}

#logo-img:hover {
  transform: scale(1.1);
}

#app-content {
  @apply flex-1 w-full pt-5;
}

#footer-container {
  @apply flex-initial w-full flex justify-between items-center px-6 py-2;
  @apply bg-darkgrey text-white border-t border-white text-xs;
}

@media only screen and (max-height: 1024px) {
  #logo-route {
  }
}

@media only screen and (max-width: 1024px) {
  #logo-route {
  }
  #footer-container {
    @apply px-2;
  }
}
</style>
