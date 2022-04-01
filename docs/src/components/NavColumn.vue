<template lang="pug">
// left side for navigation
nav#nav-tree
  // navigation tree
  div(v-for='nav in navTree').flex.flex-col.items-end
    a.nav-link(
      :key='nav.path'
      :title='nav.path'
      :href='nav.path'
      :class='{ "nav-link-active": nav.active }'
    ) {{ nav.path }}
    // when active, each entry has a nested-tree to H2 headers
    div(v-show='nav.headerInfo.length').flex.flex-col.items-end.py-2
      a.nested-link(
        v-for='header in nav.headerInfo'
        :key='header.path'
        :title='header.title'
        :href='header.path'
      ) {{ header.title }}
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
})
const contentCol = document.getElementById('content-col')
const headers = contentCol.querySelectorAll('h1, h2')
const navTree = computed(() => {
  const tree = []
  props.navPaths.forEach((path) => {
    const isActive = path === props.currentPath
    console.log(isActive)
    const headerInfo = []
    if (isActive) {
      headers.forEach((header) => {
        headerInfo.push({
          title: header.outerText,
          level: header.localName,
          id: header.id,
          path: `#${header.id}`,
        })
      })
    }
    tree.push({
      active: isActive,
      path,
      headerInfo,
    })
  })
  return tree
})

const isWide = useMediaQuery('(min-width: 750px)')
const isTall = useMediaQuery('(min-height: 620px)')

const laggedElem = null
const fillerElem = null
const scrollToTopVisible = false
const scrollTo = (targetEl) => {
  window.scroll(targetEl)
}
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
.nav-link {
  @apply text-base text-right text-theme font-medium p-2 leading-none transition-all border-b;
  &:hover {
    @apply bg-dark-grey -translate-x-1 border-light-grey;
  }
}
.nav-link-active {
  @apply bg-dark-grey -translate-x-1 border-b border-light-grey;
}
.nested-link {
  @apply text-sm text-right font-light py-1 pr-3 border-theme transition-all;
  &:hover,
  &:active {
    @apply border-r-2;
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
