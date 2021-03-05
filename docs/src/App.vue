<template lang="pug">
// don't use id# because overriden (#app) from elsewhere
main.app-container
  // left-hand side of page is navigation column
  aside#nav-column
    // split into two
    section#nav-side-by-side
      // left side for navigation
      nav#nav-tree(ref='navView')
        // logo serves as home button
        g-link#logo-container.foreground-pulse(
          to='/'
          :class='{ "pointer-events-none": $route.path === "/" }'
          title='home'
        )
          g-image#logo-img.foreground-pulse(
            src='./assets/logos/cityseer_logo_light_red.png'
            alt='logo'
            quality='90'
          )
        // go button
        g-link#go-box(v-show='isHome' to='/intro/')
          #go-button.foreground-pulse(title='Get Started!')
            font-awesome(icon='rocket' :size='smallMode ? "lg" : "2x"')
          .text-2xl.font-normal.py-2 Get Started!
        // navigation tree
        div(v-show='!isHome' v-for='doc in docNav')
          transition-group#nested-nav-tree
            g-link.nav-link(
              :key='doc.id'
              :title='doc.title'
              :to='doc.path'
              :class='{ "nav-link-active": doc.active }'
            ) {{ doc.title }}
            ClientOnly
              // when active, each entry has a nested-tree to H2 headers
              g-link.nested-link(
                v-for='h2 in doc.children'
                :key='h2.ref'
                :title='h2.value'
                :to='h2.path'
                :id='h2.ref'
              ) {{ h2.value }}
          // spacer under nested elements for active tab
          .pb-2(v-show='doc.active')
        // footer
        .flex-grow
        // github link
        a.self-start.px-6.py-1(
          href='https://github.com/benchmark-urbanism/cityseer-api'
          target='_blank'
        )
          font-awesome(:icon='["fab", "github"]' size='2x')
        footer#footer-container
          div Copyright © 2018-present Gareth Simons
      // right narrow bit for title
      g-link#title(to='/' :class='{ "pointer-events-none": $route.path === "/" }')
        h2#title-text {{ $static.metadata.siteName }}
  // right side of page is content
  #content-column(ref='routerView')
    router-view
  // back to top button
  button#back-to-top.foreground-pulse(v-show='scrollToTopVisible' @click='scrollTop()')
    font-awesome(icon='angle-double-up' size='lg')
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
      title: 'Welcome',
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
            'cityseer, urban, metrics, analytics, big data, predictive analytics, urban design, planning, property development, machine learning, API',
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
      scrollToTopVisible: false,
      largeSize: 450,
      smallSize: 250,
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
    isHome() {
      return this.$route.path === '/'
    },
    docs() {
      return this.$static.docs.edges.map((edge) => edge.node)
    },
    docNav() {
      if (!this.docs) return null
      // collect the documents in sorted order
      const sortedDocs = {}
      // iterate the paths in order
      this.navPaths.forEach((navPath) => {
        const thisDoc = this.docs.filter((doc) => doc.path === navPath).pop()
        // note if a document is active
        // catch URLs with extra trailing slash when checking for active paths
        let routerPath = this.$route.path
        if (routerPath[routerPath.length - 1] !== '/') {
          routerPath = routerPath + '/'
        }
        const isActive = thisDoc.path === routerPath
        // and if so, collect the h2 children
        const children = []
        if (isActive && process.isClient) {
          thisDoc.headings.forEach((h2) => {
            children.push({
              path: `${routerPath}${h2.anchor}`,
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
      // update logo size
      this.updateLogoSize()
    },
    docNav: {
      immediate: true,
      handler(newDocNav) {
        if (!newDocNav) return
        if (!process.isClient) return
        // update logo size
        this.updateLogoSize()
        // trigger the h2 headings animation
        this.h2Anim()
        // reset h2 element state for interactive nav
        // find the active route / tab
        const activeNav = Object.values(newDocNav)
          .filter((doc) => doc.active)
          .pop()
        if (!activeNav) return
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
    // update scroll position
    document.addEventListener('scroll', () => {
      if (window) this.scrollToTopVisible = window.scrollY > window.innerHeight
    })
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
    scrollTop() {
      window.scrollTo({ top: 0, behavior: 'smooth' })
    },
    updateLogoSize() {
      setTimeout(() => {
        if (this.isHome && !this.smallMode) {
          this.animLogoLarge()
        } else {
          this.animLogoSmall()
        }
      }, 50)
    },
    animLogoSmall() {
      anime({
        targets: '#logo-container',
        width: this.smallSize,
        'margin-top': 100,
        'margin-right': 15,
        'margin-bottom': 20,
        'margin-left': 15,
        duration: 300,
        easing: 'easeOutExpo',
      })
    },
    animLogoLarge() {
      anime({
        targets: '#logo-container',
        width: this.largeSize,
        'margin-top': 200,
        'margin-right': 50,
        'margin-bottom': 200,
        'margin-left': 50,
        duration: 500,
        easing: 'easeOutElastic',
      })
    },
    h2Anim() {
      this.$nextTick(() => {
        anime({
          targets: '.nested-link',
          opacity: [0.75, 1],
          scale: [0.98, 1],
          duration: 100,
          delay: anime.stagger(10),
          easing: 'linear',
        })
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
        if (!elem) return
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

<style lang="postcss" scoped>
.app-container {
  @apply min-h-screen w-screen flex;
}

#go-box {
  @apply w-full flex flex-col items-center justify-center;
}

#go-box:hover {
  transform: scale(1.05);
}

#go-button {
  @apply w-20 h-20 flex items-center justify-center transition-all;
  @apply border-2 border-theme bg-lightgrey rounded-full shadow text-theme;
}

#nav-side-by-side {
  @apply flex flex-grow;
}

#nav-column {
  @apply flex flex-col sticky top-0 min-h-screen max-h-screen overflow-y-auto;

  min-width: 350px;
}

#nav-tree {
  @apply flex-grow w-full flex flex-col items-end;

  background-color: rgba(211, 47, 47, 0.075);
}

#nested-nav-tree {
  @apply flex flex-col items-end;
}

#logo-container {
  /* width and margins set from animations */
  @apply w-full flex items-end;

  width: 250px;
}

#logo-img {
  @apply transition-all object-contain;
}

#logo-img:hover {
  transform: scale(1.05);
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
  @apply text-sm font-light py-0.5 pr-6 cursor-pointer;
  @apply border-theme;
}

.nested-link-active,
.nested-link:hover,
.nested-link:active {
  @apply border-r-3 pr-5 font-normal;
}

#footer-container {
  @apply self-start text-xxs px-6 py-1;
}

#title {
  @apply py-4 border-l border-theme transition-all;

  writing-mode: vertical-lr;
}

#title-text {
  @apply text-right font-light text-3xl;
}

#content-column {
  @apply flex-1 min-w-0 max-w-3xl px-6 pb-20;
}

#back-to-top {
  @apply fixed flex items-center justify-center z-50 top-4 right-4;
  @apply w-12 h-12 rounded-full border-2 border-white shadow;
  @apply bg-theme text-lightgrey;
}

@media only screen and (max-width: 958px) {
  .app-container {
    @apply flex-col items-center;
  }
  #go-box {
    @apply py-20;
  }
  #go-button {
    @apply w-16 h-16;
  }

  #nav-column {
    @apply w-full min-w-0 relative min-h-0 max-h-full overflow-visible items-start;
    @apply pl-0 border-r-0 border-b border-theme;

    min-height: 50vh;
  }
  #nav-side-by-side {
    @apply w-full;
  }
  #nav-tree {
    @apply flex-auto items-end pr-0;
  }
  #title {
    @apply flex-1 py-4 ml-0;
  }
  #title-text {
    @apply text-2xl;
  }
  .nav-link {
    @apply text-sm text-left;
  }
  .nested-link {
    @apply text-xs;
  }
  #footer-container {
    @apply text-xxs w-full self-start text-left;
  }
  #content-column {
    @apply max-w-full;
  }
}
</style>
