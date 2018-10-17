<template>
    <div>
        <v-card-text class="theme-color-text body-1">
            <div class="pt-4">

                <div class="subheading theme-color-intense light-background pa-4">
                    {{ signature }}
                </div>

                <div class="pa-2">
                    <p v-html="intro"></p>
                </div>
            </div>

            <div class="body-2 pt-3 pb-1 mb-2 border-bottom">Parameters</div>

            <div v-for="p in params" class="py-2">
                <v-layout class="row wrap align-center">
                    <v-flex class="xs12 md4">
                        <v-layout class="row wrap">
                            <v-flex class="xs12 subheading theme-color-intense">
                                $\verb!{{ p.name }}!$
                            </v-flex>
                            <v-flex class="xs12 caption">
                                {{ p.type }}
                            </v-flex>
                        </v-layout>
                    </v-flex>
                    <v-flex class="xs12 md8">
                        <p v-html="p.desc"></p>
                    </v-flex>
                </v-layout>
            </div>

            <div class="body-2 pt-3 pb-1 mb-2 border-bottom">Returns</div>

            <div v-for="r in returns" class="py-2">
                <v-layout class="row wrap align-center">
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
                    <v-flex class="xs12 md8">
                        <div v-html="r.desc"></div>
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

  export default {
    name: 'DisplayFunction',
    components: {
      VDivider,
      VLayout,
      VFlex,
      VCard,
      VCardText,
      VCardTitle
    },
    mounted () {
      window.renderMathInElement(this.$el, {
        delimiters: [
          {left: "$", right: "$", display:false}
        ]
      })
    },
    props: {
      func: {
        type: Object,
        required: true
      }
    },
    computed: {
      name () {
        return this.func['name']
      },
      intro () {
        return this.func['intro']
      },
      params () {
        return this.func['params']
      },
      returns () {
        return this.func['returns']
      },
      signature () {
        let par = []
        this.params.forEach(p => {
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
        return '$\\verb!' + this.name + '!$$\\big(' + param_str + '\\big)$'
      }
    }
  }
</script>