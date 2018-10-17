<template>
    <v-chip small label :color="color" text-color="white" class="ma-1">
        {{ text }}
    </v-chip>
</template>

<style lang="stylus" scoped>
    @import '~vuetify/src/stylus/app'
</style>

<script>
  import { VChip } from 'vuetify/lib'

  export default {
    name: 'Chip',
    components: {
      VChip
    },
    props: {
      important: {
        type: Boolean,
        required: false,
        default: false
      },
      text: {
        type: String,
        required: true,
        default: 'txt'
      }
    },
    computed: {
      color () {
        return this.important ? '#d32f2f' : '#0064b7'
      }
    }
  }
</script>