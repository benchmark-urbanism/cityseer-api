<template>
    <div>
        <v-card color="blue" class="white--text pa-2">
            <v-card-title class="pt-5 subheading">
                {{ signature }}
            </v-card-title>
            <v-card-text>
                <div title>{{ intro }}</div>
                <v-divider></v-divider>
                <li v-for="p in params">
                    {{ p.name }}
                    {{ p.type }}
                    {{ p.desc }}
                </li>
                <li v-for="r in returns">
                    {{ r.name }}
                    {{ r.type }}
                    {{ r.desc }}
                </li>
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
      },
      intro: {
        type: String,
        required: false,
        default: ''
      },
      params: {
        type: Array,
        required: false,
        default: []
      },
      returns: {
        type: Array,
        required: false,
        default: []
      }
    },
    computed: {
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