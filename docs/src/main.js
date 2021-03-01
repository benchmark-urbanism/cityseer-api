import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { config, library } from '@fortawesome/fontawesome-svg-core'
import { faArrowRight, faArrowLeft, faHome } from '@fortawesome/free-solid-svg-icons'
import '@fortawesome/fontawesome-svg-core/styles.css'
import Vuex from 'vuex'
import { format } from 'date-fns'

import './assets/css/tailwind.css'
import DefaultLayout from '~/layouts/Default.vue'

// setup fontAwesome
config.autoAddCss = false
library.add(faArrowRight, faArrowLeft, faHome)

export default function (Vue, { appOptions, head }) {
  // Set default layout as a global component
  Vue.component('Layout', DefaultLayout)
  Vue.component('FontAwesome', FontAwesomeIcon)
  head.link.push({ rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' })
  head.link.push({ rel: 'preconnect', href: 'https://fonts.gstatic.com' })
  head.link.push({
    rel: 'stylesheet',
    href:
      'https://fonts.googleapis.com/css2?family=Raleway:ital,wght@0,200;0,300;0,400;0,500;0,600;0,700;1,300&display=swap',
  })
  head.link.push({
    rel: 'stylesheet',
    href:
      'https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@200;300;400;500;600;700&display=swap',
  })
  head.link.push({
    rel: 'stylesheet',
    href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.11.1/katex.min.css',
  })
  head.link.push({
    rel: 'stylesheet',
    href:
      'https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/4.0.0/github-markdown.min.css',
  })
  // add the store
  Vue.use(Vuex)
  appOptions.store = new Vuex.Store({
    state: {
      innerWidth: null,
      innerHeight: null,
      dayId: null,
    },
    mutations: {
      setDomDims(state, { width, height }) {
        state.innerWidth = width
        state.innerHeight = height
      },
    },
    getters: {
      smallMode: (state) => {
        if (state.innerWidth && state.innerHeight) {
          return state.innerWidth < 1024
        }
        return false
      },
      makeMonthDate: () => (rawDate) => {
        if (rawDate) return format(new Date(rawDate), 'LLL y')
        return ''
      },
      makeDayDate: () => (rawDate) => {
        if (rawDate) return format(new Date(rawDate), 'do LLL, y')
        return ''
      },
    },
  })
}
