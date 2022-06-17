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
    div(
      v-show='nav.headerInfo.length'
      @click='headingTargetsAnim()').flex.flex-col.items-end.py-2
      a.nested-link(
        v-for='header in nav.headerInfo'
        :title='header.title'
        :href='header.targetPath'
        :id='header.headerId'
      ) {{ header.title }}
</template>

<script setup>
import { useIntersectionObserver, useTimeoutFn } from '@vueuse/core'
import anime from 'animejs/lib/anime.es'
import { nextTick, onMounted, reactive } from 'vue'

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
const navTree = reactive([])
let activeNavElems = []
onMounted(() => {
  const contentCol = document.getElementById('content-col')
  const headers = contentCol.querySelectorAll('h2') // h1?
  props.navPaths.forEach((path) => {
    const isActive = path === props.currentPath
    const headerInfo = []
    if (isActive) {
      headers.forEach((header) => {
        if (header.textContent.includes('__init__')) return
        const headerId = `head-${header.id}`
        const targetPath = `#${header.id}`
        headerInfo.push({
          title: header.outerText,
          headerId,
          targetPath,
        })
        useIntersectionObserver(header, (entries) => {
          const entry = entries.pop()
          const navElem = document.getElementById(headerId)
          const targetIdx = activeNavElems.findIndex((el) => el.name === navElem.id)
          // if heading visible, add to list of nav targets if properly new
          if (entry.isIntersecting && targetIdx === -1) {
            navElem.classList.add('nested-link-visible')
            activeNavElems.push({
              name: navElem.id,
              targetElem: navElem,
              visible: true,
            })
          } else if (!entry.isIntersecting && targetIdx >= 0) {
            // if target no longer visible, leave for now, but update visibility accordingly
            activeNavElems[targetIdx].visible = false
          }
          // filter out still visible targets
          const stillVisible = activeNavElems.filter((el) => el.visible)
          // if visible elements - use only those
          if (stillVisible.length) {
            activeNavElems.forEach((el) => {
              if (!el.visible) el.targetElem.classList.remove('nested-link-visible')
            })
            // turn off the rest and forget
            activeNavElems = stillVisible
          } else if (targetIdx >= 0) {
            // otherwise, keep the current element for now
            activeNavElems = [activeNavElems[targetIdx]]
          }
        })
      })
    }
    navTree.push({
      active: isActive,
      path,
      headerInfo,
    })
  })
  useTimeoutFn(() => {
    headingTargetsAnim()
  }, 100)
})
const headingTargetsAnim = () => {
  nextTick(() => {
    anime({
      targets: '.nested-link',
      scale: [0.95, 1],
      duration: 50,
      delay: anime.stagger(5),
    })
  })
}
</script>

<style lang="postcss" scoped>
.nav-link {
  @apply border-b px-2 py-2 text-right text-sm font-medium leading-none text-theme transition-all;
  &:hover {
    @apply -translate-x-1 border-light-grey bg-dark-grey;
  }
}
.nav-link-active {
  @apply -translate-x-1 border-b border-light-grey bg-dark-grey;
}
.nested-link {
  @apply border-theme py-1 pr-3 text-right text-xs font-light text-lighter-grey transition-all;
  &:hover,
  &:active {
    @apply border-r-2;
  }
}
.nested-link-visible {
  @apply border-r-2;
}
@media only screen and (max-width: 958px) {
  .nav-link {
    @apply text-left text-sm;
  }
  .nested-link {
    @apply text-xs;
  }
}
</style>
