<template lang="pug">
// left side for navigation
nav#nav-tree
  // navigation tree
  div(v-for='nav in navTree').nested-nav-tree
    a.nav-link(
      :key='nav.path'
      :title='nav.path'
      :href='nav.path'
      :class='{ "nav-link-active": nav.active }'
    ) {{ nav.path }}
    // when active, each entry has a nested-tree to H2 headers
    a.nested-link(
      v-for='header in nav.headers'
      :key='header.path'
      :title='header.title'
      :href='header.path'
    ) {{ header.title }}
    // spacer under nested elements for active tab
    .pb-2(v-show='nav.active')
</template>

<script setup>
import { useIntersectionObserver, useThrottleFn, useMediaQuery } from '@vueuse/core'
import { computed, ref } from 'vue'
import { Rocket, Github } from '@vicons/fa'

const props = defineProps({
  navPaths: {
    type: Array,
    required: true,
  },
  currentPath: {
    type: String,
    required: true,
  },
  pageHeaders: {
    type: Object,
    required: true,
  },
})
const navTree = computed(() => {
  const tree = []
  if (!props.navPaths) return tree
  props.navPaths.forEach((path) => {
    const isActive = path === props.currentPath
    const headers = []
    if (isActive) {
      props.pageHeaders.forEach((header) => {
        console.log(header)
        headers.push({
          title: header.slug.replace('-', ' '),
          path: `#${header.slug}`,
        })
      })
    }
    tree.push({
      active: isActive,
      path,
      headers,
    })
  })
  return tree
})

const isWide = useMediaQuery('(min-width: 750px)')
const isTall = useMediaQuery('(min-height: 620px)')

const target = ref(null)

const observerReady = false
const h2Elems = {}
const laggedElem = null
const fillerElem = null
const scrollToTopVisible = false

const scrollTo = computed((targetEl) => {
  window.scroll(targetEl)
})
const onElementObserved = (entries) => {
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
}
const observer = new IntersectionObserver(onElementObserved)
// this.$refs.routerView.querySelectorAll('h2').forEach((el) => {
//   this.observer.observe(el)
// })
</script>

<style lang="postcss" scoped>
.nested-nav-tree {
  @apply flex flex-col items-end;
}
#logo-img {
  @apply w-full object-contain transition-all;
}
.nav-link {
  @apply text-base text-right text-theme font-medium px-3 py-3 cursor-pointer leading-none;
  & :hover,
  & :active {
    @apply bg-theme text-white;
  }
}
#logo-link {
  /* width and margins set from animations */
  @apply w-full flex items-center justify-end transition-all;

  & :hover {
    transform: scale(1.05);
  }
}
.nav-link-active {
  @apply border-b border-t border-light-grey border-r-3 pr-2;
}
.nested-link {
  @apply text-xs font-light py-1 pr-3 cursor-pointer;
  @apply border-theme;
  &:hover,
  &:active {
    @apply border-r-3 pr-2;
  }
}
@media only screen and (max-width: 958px) {
  .nav-link {
    @apply text-sm text-left;
  }
  .nested-link {
    @apply text-xs;
  }
}
</style>
