<template lang="pug">
// don't use id# because overriden (#app) from elsewhere
main.app-container
  aside#nav-column
    header
      router-link#logo-route(
        to='/'
        :class='{ "pointer-events-none": $route.path === "/" }'
        title='home'
      )
        img#logo-img.foreground-pulse(src='./assets/logos/cityseer_logo_light_red.png' alt='logo')
    section.flex.flex-grow
      title#title {{ $static.metadata.siteName }}
      nav#nav-container
        router-link.nav-link.self-center.p-6(to='/' title='to home')
          // figure out how to inject link SVG...?
          span.svg-inline--fa.fa-link.fa-w-18.fa-lg.text-theme
          font-awesome.text-theme(icon='home' size='lg')
        div(v-for='doc in docNav' :key='doc.id' @click='animChildren(doc.path)')
          .nav-link(:class='{ "nav-link-active": doc.active }') {{ doc.title }}
          .nested-link(v-for='h2 in doc.children' :key='h2.anchor' @click='scrollTo(h2.anchor)') {{ h2.value }}
          .pb-2(v-if='doc.active')
    footer#footer-container
      div Copyright © 2018-present Gareth Simons
  #content-column.overflow-y-auto(ref='routerView')
    router-view.max-w-3xl
</template>

<static-query>
query {
  metadata {
    siteName
    siteDescription
    siteUrl
    pathPrefix
  },
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
        headings (depth: h2) {
          depth
          anchor
          value
        }
        content
      }
    }
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
      observer: null,
      logoLarge: true,
      largeSize: 250,
      smallSize: 100,
      navPaths: [
        '/intro/',
        '/guide/cleaning/',
        '/tools/graphs/',
        '/tools/plot/',
        '/tools/mock/',
        '/metrics/layers/',
        '/metrics/networks/',
        '/attribution/',
      ],
    }
  },
  computed: {
    ...mapGetters(['smallMode']),
    docs() {
      return this.$static.docs.edges.map((edge) => edge.node)
    },
    docNav() {
      if (!this.docs) return null
      const sortedDocs = {}
      this.navPaths.forEach((navPath) => {
        const thisDoc = this.docs.filter((doc) => doc.path == navPath).pop()
        const isActive = thisDoc.path === this.$route.path
        sortedDocs[navPath] = {
          id: thisDoc.id,
          path: thisDoc.path,
          title: thisDoc.title,
          active: isActive,
          children: isActive ? thisDoc.headings : [],
        }
      })
      console.log(sortedDocs)
      console.log(this.$route.path, sortedDocs)
      return sortedDocs
    },
  },
  watch: {
    smallMode() {
      if (!process.isClient) return
      this.updateLogoSize()
    },
    $route() {
      console.log(this.$route, this.docNav)
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

    // active scroll
    this.observer = new IntersectionObserver(this.onElementObserved, {
      root: this.$refs.scrollable,
      rootMargin: '0px',
      theshold: 1.0,
    })
    // this.$refs.scrollable.querySelectorAll('h2').forEach((el) => {
    //   this.observer.observe(el)
    // })
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
    scrollTo(targetEl) {
      this.$router.push(targetEl)
    },
    updateLogoSize() {
      if (!this.smallMode && !this.logoLarge) {
        this.animLogoLarge()
      } else if (this.smallMode && this.logoLarge) {
        this.animLogoSmall()
      }
    },
    animChildren(targetPath) {
      this.$router.push(targetPath, () => {
        this.$nextTick(() => {
          anime({
            targets: '.nested-link',
            scaleY: 0,
            opacity: 0,
            duration: 0,
            complete() {
              anime({
                targets: '.nested-link',
                scaleY: 1,
                opacity: 1,
                duration: 40,
                delay: anime.stagger(20),
                easing: 'easeOutSine',
              })
            },
          })
        })
      })
    },
    animLogoSmall() {
      anime({
        targets: '#logo-route',
        width: this.smallSize,
        duration: 500,
        complete: () => {
          this.logoLarge = false
        },
      })
    },
    animLogoLarge() {
      anime({
        targets: '#logo-route',
        width: this.largeSize,
        duration: 500,
        complete: () => {
          this.logoLarge = true
        },
      })
    },
    onElementObserved(entries) {
      entries.forEach(({ target, isIntersecting }) => {
        console.log(target, isIntersecting)
      })
    },
  },
}
</script>

<style lang="postcss">
.fade-enter,
.fade-leave-to {
  transform: scaleY(0);
}
.fade-enter-to,
.fade-leave {
  transform: scaleY(1);
}
.fade-enter-active {
  transition: all 250ms ease-out;
}
.fade-leave-active {
  transition: all 0s linear;
}
</style>

<style lang="postcss" scoped>
.app-container {
  @apply h-screen w-screen grid gap-0;

  grid-template-columns: 30% 70%;
}

#nav-column {
  @apply col-start-1 col-span-1 sticky top-0 w-full max-h-screen;
  @apply flex flex-col text-right;
}

#content-column {
  @apply col-start-2 col-span-1;
}

#logo-route {
  @apply relative flex items-end m-4 self-end;

  width: 250px;
}

#logo-img {
  @apply transition-all w-full object-contain;
}

#logo-img:hover {
  transform: scale(1.1);
}

#title-container {
  @apply flex mx-0.5 bg-blue-100;
}

#title {
  @apply px-3 py-8;
  @apply text-6xl tracking-tight font-extralight;

  text-shadow: 0 -0.5px 1px #fff;
  writing-mode: vertical-lr;
}

#nav-container {
  @apply w-full pr-8;
}

.nav-link {
  @apply text-lg text-theme font-normal py-1 px-4 cursor-pointer;
}

.nav-link-active,
.nav-link:hover {
  @apply bg-theme text-white;
}

.nested-link {
  @apply text-sm font-light py-0.5 pr-4 cursor-pointer;
  @apply border-r border-theme;
}

.nested-link-active,
.nested-link:hover {
  @apply bg-darkgrey text-lightgrey;
}

#app-content {
  @apply flex-1 w-full pt-5;
}

#footer-container {
  @apply w-full flex justify-between items-center text-xs p-2;
}

@media only screen and (max-height: 958px) {
  #logo-route {
  }
}

@media only screen and (max-width: 958px) {
  #side-bar {
    @apply w-0;
  }
  #logo-route {
  }
  #footer-container {
    @apply px-2;
  }
}
</style>
