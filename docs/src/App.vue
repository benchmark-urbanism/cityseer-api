<template lang="pug">
// don't use id# because overriden (#app) from elsewhere
main.app-container
  // left-hand side of page is navigation column
  aside#nav-column
    // split into two
    section.flex-grow.flex
      // left narrow bit for title
      h1#title {{ $static.metadata.siteName }}
      // right bit for navigation
      nav#nav-container(ref='navView')
        // logo serves as home button
        router-link#logo-route.foreground-pulse(
          to='/'
          :class='{ "pointer-events-none": $route.path === "/" }'
          title='home'
        )
          img#logo-img.foreground-pulse(
            src='./assets/logos/cityseer_logo_light_red.png'
            alt='logo'
          )
        // navigation tree
        .flex.flex-col.items-end(v-for='doc in docNav' :key='doc.id')
          // each entry has a button
          button.nav-link(
            :title='doc.title'
            @click.self='toNavPathAnim(doc.path)'
            :class='{ "nav-link-active": doc.active }'
          ) {{ doc.title }}
          // when active, each entry has a nested-tree to H2 headers
          router-link.nested-link.pb-2(
            v-for='h2 in doc.children'
            :key='h2.ref'
            :to='h2.anchor'
            :title='h2.value'
            :id='h2.ref'
          ) {{ h2.value }}
    // footer
    footer#footer-container
      div Copyright © 2018-present Gareth Simons
  // right side of page is content
  section#content-column(ref='routerView')
    router-view
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
      observerReady: false,
      h2Elems: {},
      laggedElem: null,
      fillerElem: null,
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
      // collect the documents in sorted order
      const sortedDocs = {}
      // iterate the paths in order
      this.navPaths.forEach((navPath) => {
        const thisDoc = this.docs.filter((doc) => doc.path == navPath).pop()
        // note if a document is active
        const isActive = thisDoc.path === this.$route.path
        // and if so, collect the h2 children
        const children = []
        if (isActive) {
          thisDoc.headings.forEach((h2) => {
            children.push({
              anchor: h2.anchor,
              value: h2.value,
              ref: this.makeElemId(h2.anchor),
              fullRef: this.makeElemId(h2.anchor, true),
            })
          })
        }
        sortedDocs[navPath] = {
          id: thisDoc.id,
          path: thisDoc.path,
          title: thisDoc.title,
          active: isActive,
          children,
        }
      })
      return sortedDocs
    },
  },
  watch: {
    smallMode() {
      if (!process.isClient) return
      this.updateLogoSize()
    },
    docNav: {
      immediate: true,
      handler(newDocNav) {
        // reset h2 element state for interactive nav
        // find the active route / tab
        const activeNav = Object.values(newDocNav)
          .filter((doc) => doc.active)
          .pop()
        // setup the state dictionary
        this.h2Elems = {}
        activeNav.children.forEach((h2) => {
          this.h2Elems[h2.fullRef] = false
        })
        // lagged and filler elements also need resetting
        this.laggedElem = null
        this.fillerElem = null
        // reset observer
        this.observerReady = false
        if (this.observer) {
          this.observer.disconnect()
          this.observer = null
        }
        // use a timeout for setting new intersection observer, otherwise elements may not exist
        setTimeout(() => {
          this.observer = new IntersectionObserver(this.onElementObserved)
          this.$refs.routerView.querySelectorAll('h2').forEach((el) => {
            this.observer.observe(el)
          })
          // and use a flag to prevent unexpected async issues
          this.observerReady = true
        }, 100)
      },
    },
  },
  created() {
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
    toNavPathAnim(targetPath) {
      this.$router.push(targetPath, () => {
        this.$nextTick(() => {
          anime({
            targets: '.nested-link',
            opacity: 0,
            scale: 0.95,
            duration: 0,
            complete() {
              anime({
                targets: '.nested-link',
                opacity: 1,
                scale: [0.95, 1],
                duration: 150,
                delay: anime.stagger(15),
                easing: 'easeInQuint',
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
    makeElemId(anchor, asSelector = false) {
      const body = this.$route.path.replaceAll('/', '-')
      const cleanAnchor = anchor.replaceAll('#', '')
      let tag = `io-id-${body}${cleanAnchor}`.replace('--', '-')
      if (asSelector) {
        tag = '#' + tag
      }
      return tag
    },
    onElementObserved(entries) {
      if (!this.observerReady) return
      if (!('navView' in this.$refs) || !this.$refs.navView) return
      // target class
      const targetClass = 'nested-link-active'
      // clean up fillers - these will be re-enabled if necessary
      if (this.fillerElem) {
        this.fillerElem.classList.remove(targetClass)
        this.fillerElem = null
      }
      // process state changes
      entries.forEach((entry) => {
        const elemId = this.makeElemId(entry.target.id, true)
        const elem = this.$refs.navView.querySelector(elemId)
        // add new
        if (entry.isIntersecting) {
          elem.classList.add(targetClass)
          // refresh lagged element to most recent
          this.laggedElem = elem
          // and update state
          this.h2Elems[elemId] = true
        } else {
          // remove old
          elem.classList.remove(targetClass)
          // and update state
          this.h2Elems[elemId] = false
        }
      })
      // count the existing active instances - an object is necessary for tracking state
      const countActive = Object.values(this.h2Elems).reduce((aggState, currentState) => {
        if (currentState) return (aggState += 1)
        return aggState
      }, 0)
      // if there are no existing or new active elements, use the trailing element as a filler
      if (!countActive && this.laggedElem) {
        this.fillerElem = this.laggedElem
        this.fillerElem.classList.add(targetClass)
      }
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

.foreground-pulse {
  opacity: 1;
  animation-name: foreground-pulsate;
  animation-duration: 0.75s;
  animation-direction: alternate;
  animation-iteration-count: infinite;
  animation-timing-function: ease-in-out;
}

@keyframes foreground-pulsate {
  0% {
    opacity: 1;
  }

  100% {
    opacity: 0.95;
  }
}
</style>

<style lang="postcss" scoped>
.app-container {
  @apply min-h-screen w-screen flex;
}

#nav-column {
  @apply sticky top-0 h-screen bg-red-50 overflow-y-auto;
  @apply flex flex-col items-end text-right border-r border-midgrey;
}

#content-column {
  @apply max-w-3xl px-6;
}

#logo-route {
  @apply w-full flex items-end py-10;

  width: 250px;
}

#logo-img {
  @apply transition-all w-full object-contain;
}

#logo-img:hover {
  transform: scale(1.1);
}

#title-container {
  @apply flex mx-0.5 bg-blue-500;
}

#title {
  @apply pr-4 py-8;
  @apply text-4xl tracking-tight leading-normal font-extralight;

  text-shadow: 0 -0.5px 1px #fff;
  writing-mode: vertical-lr;
}

#footer-container {
  @apply w-full text-right text-xs px-4 py-1;
}

#nav-container {
  @apply w-full pr-4 flex flex-col;
}

.nav-link {
  @apply text-base text-right text-theme font-medium px-3 py-1 cursor-pointer;
}

.nav-link-active,
.nav-link:hover,
.nav-link:active {
  @apply bg-theme text-white;
}

.nested-link {
  @apply text-sm font-light py-0.5 pr-4 cursor-pointer;
  @apply border-r border-theme;
}

.nested-link-active,
.nested-link:hover,
.nested-link:active {
  @apply border-r-3 font-medium pr-3;
}

#app-content {
  @apply flex-1 w-full pt-5;
}

@media only screen and (max-height: 958px), (max-width: 958px) {
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
