import Vuex from 'vuex'
// font awesome
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { config, library } from '@fortawesome/fontawesome-svg-core'
import { faHome, faLink, faAngleDoubleUp, faRocket } from '@fortawesome/free-solid-svg-icons'
import { faGithub } from '@fortawesome/free-brands-svg-icons'
import '@fortawesome/fontawesome-svg-core/styles.css'
// tailwind
import './assets/css/tailwind.css'
// prism
import 'prismjs/themes/prism-okaidia.css'
// default layout
import DefaultLayout from '~/layouts/Default.vue'
// import custom components
import Chip from './components/Chip.vue'
import FuncElement from './components/FuncElement.vue'
import FuncHeading from './components/FuncHeading.vue'
import FuncSignature from './components/FuncSignature.vue'

// setup fontAwesome
config.autoAddCss = false
library.add(faHome, faLink, faAngleDoubleUp, faRocket, faGithub)

export default function (Vue, { appOptions, head }) {
  // Set default layout as a global component
  Vue.component('Layout', DefaultLayout)
  // Add global components for rendering markdown
  Vue.component('Chip', Chip)
  Vue.component('FuncElement', FuncElement)
  Vue.component('FuncHeading', FuncHeading)
  Vue.component('FuncSignature', FuncSignature)
  // Font Awesome
  Vue.component('FontAwesome', FontAwesomeIcon)
  // Head
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
        if (state.innerWidth) {
          return state.innerWidth < 958
        }
        return false
      },
    },
  })
}
