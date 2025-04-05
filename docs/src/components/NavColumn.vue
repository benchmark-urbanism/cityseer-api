<template lang="pug">
// left side for navigation
nav#nav-tree
  // navigation tree
  div(v-for='nav in navTree').flex.flex-col.items-end
    a(
      :class='{ "nav-link-active": nav.active }',
      :href='nav.path',
      :key='nav.path',
      :title='nav.path'
    ).nav-link {{ nav.path }}
    // when active, each entry has a nested-tree to H2 headers
    div(
      @click='headinganim()'
      v-show='nav.headerInfo.length'
    ).flex.flex-col.items-end.py-2
      a(
        :href='header.targetPath',
        :id='header.headerId',
        :title='header.title'
        v-for='header in nav.headerInfo'
      ).nested-link {{ header.title }}

</template>

<script setup>
import { useIntersectionObserver, useTimeoutFn }

 from '@vueuse/core'
import { animate, stagger}

 from 'animejs'
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
  useTimeoutFn(() => {
    prepareSideBarNav()
  }, 50)
})
const prepareSideBarNav = () => {
  const contentCol = document.getElementById('content-col')
  const headers = contentCol.querySelectorAll('h2') // h1?
  // enforce no trailing slash
  let currentPath = props.currentPath
  if (currentPath.slice(-1) === '/') {
    currentPath = currentPath.slice(0, -1)
  }
  props.navPaths.forEach((path) => {
    const isActive = path === currentPath
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
    headinganim()
  }, 100)
}

const headinganim = () => {
  nextTick(() => {
    animate({
      targets: '.nested-link',
      scale: [0.95, 1],
      duration: 50,
      delay: stagger(5),
    })
  })
}

</script>

<style scoped>
.nav-link {
  border-bottom: 1px solid var(--color-dark-grey);
  padding: 0.15rem 0.3rem;
  text-align: right;
  font-size: var(--text-sm);
  font-weight: var(--font-weight-medium);
  color: var(--color-theme);
  transition: all 0.3s ease;
}

.nav-link:hover {
  transform: translateX(-0.25rem);
  border-color: var(--color-light-grey);
  background-color: var(--color-dark-grey);
}

.nav-link-active {
  transform: translateX(-0.3rem);
  border-bottom: 1px solid var(--color-light-grey);
  background-color: var(--color-dark-grey);
}

.nested-link {
  border-color: var(--color-theme);
  padding: 0.1rem 0.3rem;
  text-align: right;
  font-size: var(--text-xs);
  font-weight: var(--font-weight-extralight);
  color: var(--color-lighter-grey);
  transition: all 0.3s ease;
}

.nested-link:hover,
.nested-link:active {
  border-right: 2px solid var(--color-theme);
}

.nested-link-visible {
  border-right: 2px solid var(--color-theme);
}

@media only screen and (width <= 958px) {
  .nav-link {
    text-align: left;
    font-size: var(--text-sm);
  }

  .nested-link {
    font-size: var(--text-xs);
  }
}
</style>
