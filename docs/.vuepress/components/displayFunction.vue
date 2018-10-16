<template>
    <div>
        <v-card class="pa-2 text-theme elevation-0">
            <v-card-text class="theme-color-text">

                <div class="subheading theme-color-intense pt-3">
                    {{ signature }}
                </div>

                <div class="body-1 pt-1 pb-3">{{ intro }}</div>

                <div v-for="p in params" class="py-2">
                    <v-layout class="row wrap">
                        <v-flex class="xs12 md4">
                            <v-layout class="row wrap">
                                <v-flex class="xs12 subheading theme-color-intense">
                                    $\verb!{{ p.name }}!$
                                </v-flex>
                                <v-flex class="xs12 caption">
                                    {{ p.type }}
                                </v-flex>
                                <v-flex v-if="p.def" class="xs12 caption">
                                    ={{ p.def }}
                                </v-flex>
                            </v-layout>
                        </v-flex>
                        <v-flex class="xs12 md8">
                            {{ p.desc }}
                        </v-flex>
                    </v-layout>
                </div>

                <div class="subheading py-3">Returns</div>
                <div v-for="r in returns">
                    <v-layout class="row wrap">
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
                            {{ r.desc }}
                        </v-flex>
                    </v-layout>
                </div>
            </v-card-text>
        </v-card>
    </div>
</template>

<style lang="stylus">
    @import '~vuetify/src/stylus/app'


</style>

<script>

  import { VDivider, VLayout, VFlex, VCard, VCardText, VCardTitle } from 'vuetify/lib'

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
      console.log(this.$page)
      console.log(this.$site)
      window.renderMathInElement(this.$el, {
        delimiters: [
          {left: '$', right: '$', display: false},
          {left: '$$', right: '$$', display: true},
          {left: '\\(', right: '\\)', display: false},
          {left: '\\[', right: '\\]', display: true},
        ]
      })
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
        return this.$page.frontmatter.functions[this.func]['intro']
      },
      params () {
        return this.$page.frontmatter.functions[this.func]['params']
      },
      returns () {
        return this.$page.frontmatter.functions[this.func]['returns']
      },
      signature () {
        let par = []
        this.params.forEach(p => {
          par.push(p.name)
        })
        let param_str = ''
        if (par) {
          param_str = '\\verb!' + par.join('!\,\\ \\verb!') + '!'
        }
        return '$\\verb!' + this.name + '!\\big(' + param_str + '\\big)$'
      }
    }
  }
</script>