<template lang="pug">
// split into two
section#nav-side-by-side
  // left side for navigation
  nav#nav-tree(ref='navView')
    // logo serves as home button
    a#logo-link.foreground-pulse(
      href='/'
      title='home'
    )
      a#logo-img(
        src='./assets/logos/cityseer_logo_light_red.png'
        alt='logo'
        :style='logoDynamicStyle'
      )
    // go button
    a#go-box(v-show='isHome' href='/intro/')
      #go-button.foreground-pulse(title='Get Started!')
        font-awesome(icon='rocket' :size='smallMode ? "lg" : "2x"')
      .text-2xl.font-normal.py-2 Get Started!
    // navigation tree
    div(v-show='!isHome' v-for='doc in docNav')
      transition-group#nested-nav-tree
        a.nav-link(
          :key='doc.id'
          :title='doc.title'
          :href='doc.path'
          :class='{ "nav-link-active": doc.active }'
        ) {{ doc.title }}
        ClientOnly
          // when active, each entry has a nested-tree to H2 headers
          a.nested-link(
            v-for='h2 in doc.children'
            :key='h2.ref'
            :title='h2.value'
            :href='h2.path'
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
  a#title(href='/' :class='{ "pointer-events-none": $route.path === "/" }')
    h2#title-text {{ $static.metadata.siteName }}
</template>

<script setup>
import { useIntersectionObserver, useThrottleFn, useMediaQuery } from '@vueuse/core'
import { computed, ref } from 'vue'

const isWide = useMediaQuery('(min-width: 750px)')
const isTall = useMediaQuery('(min-height: 620px)')

const target = ref(null)

const observerReady = false
const h2Elems = {}
const laggedElem = null
const fillerElem = null
const scrollToTopVisible = false
const navPaths = [
  '/intro/',
  '/guide/',
  '/examples/',
  '/tools/graphs/',
  '/tools/plot/',
  '/tools/mock/',
  '/metrics/layers/',
  '/metrics/networks/',
  '/attribution/',
]
const scrollTo = computed((targetEl) => {
  window.scroll(targetEl)
})
const observer = new IntersectionObserver(this.onElementObserved)
// this.$refs.routerView.querySelectorAll('h2').forEach((el) => {
//   this.observer.observe(el)
// })
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
</script>

<style lang="postcss" scoped></style>
