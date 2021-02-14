// .vitepress/theme/index.js
import DefaultTheme from 'vitepress/theme'
import Chip from './components/Chip.vue'
import FuncElement from './components/FuncElement.vue'
import FuncHeading from './components/FuncHeading.vue'
import FuncSignature from './components/FuncSignature.vue'
import ImageModal from './components/ImageModal.vue'
import './styles/index.css'

export default {
    ...DefaultTheme,
    enhanceApp({ app, router, siteData }) {
        app.component('Chip', Chip)
        app.component('FuncElement', FuncElement)
        app.component('FuncHeading', FuncHeading)
        app.component('FuncSignature', FuncSignature)
        app.component('ImageModal', ImageModal)
    }
}
