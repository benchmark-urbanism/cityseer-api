<template>
    <div>
        <v-layout class="row wrap param">

            <v-flex class="xs12 md4 align-self-start">
                <v-layout class="row wrap">

                    <v-flex class="xs12 subheading elem-name-style">
                        {{ name }}
                    </v-flex>

                    <v-flex class="xs12 small-text">
                        {{ type }}
                    </v-flex>

                </v-layout>
            </v-flex>

            <v-flex class="xs12 md8">
                <slot></slot>
            </v-flex>

        </v-layout>
    </div>
</template>

<style lang="stylus" scoped>

    .elem-name-style
        font-size 16px
        font-family monospace
        font-weight bold

    .small-text
        font-size 12px

    .param
        margin-top 30px
        margin-bottom 30px

    p
        margin-top 0

</style>

<script>
  // use absolute paths -> see bug https://github.com/vuejs/vuepress/issues/451
  import { VFlex, VLayout } from '../../../../node_modules/vuetify/lib'

  export default {
    name: 'DisplayFunction',
    components: {
      VLayout,
      VFlex
    },
    mounted () {
      window.renderMathInElement(this.$el, {
        delimiters: [
          {left: "$$", right: "$$", display: true}, // has to be first!!
          {left: "$", right: "$", display:false}
        ]
      })
    },
    props: {
      name: {
        type: String,
        required: true
      },
      type: {
        type: String,
        required: false,
        default: ''
      }
    }
  }
</script>