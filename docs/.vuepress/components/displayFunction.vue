<template>
    <div>
        <v-card-text class="theme-color-text body-1">

            <div class="py-3">
                <div class="subheading theme-color-intense light-background pa-3">
                    {{ signature }}
                </div>
            </div>

            <div class="pb-2">{{ intro }}</div>

            <div class="body-2 pt-4 pb-1 mb-2 border-bottom">Parameters</div>

            <div v-for="p in params">
                <v-layout class="row wrap py-1 align-center">
                    <v-flex class="xs12 md4">
                        <v-layout class="row wrap">
                            <v-flex class="xs12 subheading theme-color-intense">
                                $\verb!{{ parse(p.name) }}!$
                            </v-flex>
                            <v-flex class="xs12 caption">
                                {{ p.type }}
                            </v-flex>
                        </v-layout>
                    </v-flex>
                    <v-flex class="xs12 md8 py-2">
                        {{ p.desc }}
                    </v-flex>
                </v-layout>
            </div>

            <div class="body-2 pt-4 pb-1 mb-2 border-bottom">Returns</div>

            <div v-for="r in returns">
                <v-layout class="row wrap py-1 align-center">
                    <v-flex class="xs12 md4">
                        <v-layout class="row wrap">
                            <v-flex class="xs12 subheading theme-color-intense">
                                $\verb!{{ r.name }}!$
                            </v-flex>
                            <v-flex class="xs12 caption">
                                {{ r.type }}
                            </v-flex>
                        </v-layout>
                    </v-flex>
                    <v-flex class="xs12 md8 py-2">
                        {{ r.desc }}
                    </v-flex>
                </v-layout>
            </div>
        </v-card-text>
    </div>
</template>

<style lang="stylus" scoped>
    @import '~vuetify/src/stylus/app'
</style>

<script>

  import { VDivider, VLayout, VFlex, VCard, VCardText, VCardTitle } from 'vuetify/lib'

  const mdItMj = require('markdown-it-mathjax')()
  const md = require('markdown-it')().use(mdItMj)

  export default {
    name: 'displayFunction',
    components: {
      VDivider,
      VLayout,
      VFlex,
      VCard,
      VCardText,
      VCardTitle
    },
    mounted () {
      // window.renderMathInElement(this.$el, {
      //   delimiters: [
      //     {left: '$', right: '$', display: false},
      //     {left: '$$', right: '$$', display: true},
      //     {left: '\\(', right: '\\)', display: false},
      //     {left: '\\[', right: '\\]', display: true},
      //   ]
      // })
    },
    props: {
      func: {
        type: String,
        required: true
      }
    },
    computed: {
      name () {
        return this.func
      },
      intro () {
        return this.parse(this.$page.frontmatter.functions[this.func]['intro'])
      },
      params () {
        let par = []
        this.$page.frontmatter.functions[this.func]['params'].forEach(p => {
          let d = {}
          for (const key of Object.keys(p)) {
            d[key] = this.parse(p[key])
          }
          par.push(d)
        })
        return par
      },
      returns () {
        let ret = []
        this.$page.frontmatter.functions[this.func]['returns'].forEach(r => {
          let d = {}
          for (const key of Object.keys(r)) {
            d[key] = this.parse(r[key])
          }
          ret.push(d)
        })
        return ret
      },
      signature () {
        let par = []
        this.$page.frontmatter.functions[this.func]['params'].forEach(p => {
          if (p.def) {
            par.push(p.name + '=' + p.def)
          } else {
            par.push(p.name)
          }
        })
        let param_str = ''
        if (par) {
          param_str = '\\verb!' + par.join('!,$ $\\verb!') + '!'
        }
        param_str = '$\\verb!' + this.name + '!$$\\big(' + param_str + '\\big)$'
        return this.parse(param_str)
      }
    },
    methods: {
      parse (val) {
        let t = md.renderInline(val)
        return t
      }
    }
  }
</script>