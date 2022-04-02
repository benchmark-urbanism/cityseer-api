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
import { onMounted, nextTick, reactive } from 'vue'
import anime from 'animejs/lib/anime.es'

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
const headers = contentCol.querySelectorAll('h1, h2, h3')
const navTree = reactive([])
let activeNavElems = []
onMounted(() => {
  props.navPaths.forEach((path) => {
    const isActive = path === props.currentPath
    const headerInfo = []
    if (isActive) {
      headers.forEach((header) => {
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
      scale: [0.9, 1],
      duration: 100,
      delay: anime.stagger(10),
    })
  })
}
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
  @apply text-sm text-right font-light py-1 pr-3 border-theme transition-all text-lighter-grey;
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
    @apply text-sm text-left;
  }
  .nested-link {
    @apply text-xs;
  }
}
</style>
