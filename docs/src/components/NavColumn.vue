<template lang="pug">
// left side for navigation: all sections, active section expanded
nav#nav-tree
  div(:key='section.id', v-for='section in navTree').flex.w-full.flex-col.items-end
    a(
      :class='{ "section-header-active": section.isActive }',
      :href='section.base',
      :title='section.label'
    ).section-header {{ section.label }}
    div(v-if='section.isActive').flex.w-full.flex-col.items-end.pb-2
      div(:key='group.key', v-for='group in section.groups').flex.w-full.flex-col.items-end
        span(v-if='group.title').nav-group-title {{ group.title }}
        div(:key='nav.path', v-for='nav in group.entries').flex.w-full.flex-col.items-end
          a(
            :class='{ "nav-link-active": nav.active }',
            :href='nav.path',
            :title='nav.path'
          ).nav-link {{ nav.label }}
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
import { useIntersectionObserver, useTimeoutFn } from '@vueuse/core'
import { animate, stagger } from 'animejs'
import { nextTick, onMounted, reactive } from 'vue'

const props = defineProps({
  sections: {
    type: Array,
    required: true,
  },
  activeSectionId: {
    type: String,
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
  const headers = contentCol ? contentCol.querySelectorAll('h2') : []
  // enforce no trailing slash
  let currentPath = props.currentPath
  if (currentPath.slice(-1) === '/') {
    currentPath = currentPath.slice(0, -1)
  }
  props.sections.forEach((section) => {
    const isActive = section.id === props.activeSectionId
    const groups = []
    if (isActive) {
      section.groups.forEach((group, groupIdx) => {
        const entries = []
        group.items.forEach((item) => {
          const isCurrent = item.path === currentPath
          const headerInfo = []
          if (isCurrent) {
            headers.forEach((header) => {
              if (header.textContent.includes('__init__')) return
              const headerId = `head-${header.id}`
              const targetPath = `#${header.id}`
              headerInfo.push({
                title: header.outerText,
                headerId,
                targetPath,
              })
              useIntersectionObserver(header, (obsEntries) => {
                const entry = obsEntries.pop()
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
          entries.push({
            active: isCurrent,
            path: item.path,
            label: item.label,
            headerInfo,
          })
        })
        groups.push({
          key: `${section.id}-group-${groupIdx}`,
          title: group.title,
          entries,
        })
      })
    }
    navTree.push({
      id: section.id,
      label: section.label,
      base: section.base,
      isActive,
      groups,
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
.section-header {
  margin-top: 0.4rem;
  border-bottom: 1px solid var(--color-mid-grey);
  padding: 0.25rem 0.3rem;
  width: 100%;
  text-align: right;
  font-size: var(--text-xs);
  font-weight: var(--font-weight-medium);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-light-grey);
  transition: all 0.3s ease;
}

.section-header:hover {
  color: var(--color-lighter-grey);
  background-color: var(--color-dark-grey);
}

.section-header-active {
  color: var(--color-theme);
  border-bottom-color: var(--color-theme);
}

.nav-group-title {
  padding: 0.5rem 0.3rem 0.1rem;
  text-align: right;
  font-size: var(--text-xxs);
  font-weight: var(--font-weight-medium);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--color-light-grey);
}

.nav-link {
  border-bottom: 1px solid var(--color-dark-grey);
  padding: 0.15rem 0.3rem;
  text-align: right;
  font-size: var(--text-xs);
  font-weight: var(--font-weight-normal);
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
  font-size: var(--text-xxs);
  font-weight: var(--font-weight-extralight);
  color: var(--color-lighter-grey);
  transition: all 0.3s ease;

  /* keep long headings (e.g. the how-to questions) from flooding the nav */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  overflow: hidden;
}

.nested-link:hover,
.nested-link:active {
  border-right: 2px solid var(--color-theme);
}

.nested-link-visible {
  border-right: 2px solid var(--color-theme);
}

@media only screen and (width <= 958px) {
  .section-header,
  .nav-group-title,
  .nav-link {
    text-align: left;
    font-size: var(--text-xs);
  }

  .nested-link {
    font-size: var(--text-xxs);
  }
}
</style>
