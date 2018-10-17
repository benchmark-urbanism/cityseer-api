<template>
    <v-chip small :outline="!important" :color="color" :text-color="tColor" class="pa-0 ma-0">
        {{ text }}
    </v-chip>
</template>

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
      },
      tColor () {
        return this.important ? '#fff' : '#0064b7'
      }
    }
  }
</script>