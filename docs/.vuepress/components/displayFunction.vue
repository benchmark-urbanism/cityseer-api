<template>
    <div>
        <v-card color="blue">
            <v-card-title primary-title class="white--text">
                <slot name="function"></slot>
            </v-card-title>
        </v-card>
    </div>
</template>

<style lang="stylus">
    @import '~vuetify/src/stylus/app'
</style>

<script>
  import { VCard, VCardText, VCardTitle } from 'vuetify/lib'

  export default {
    name: 'displayFunction',
    components: {
        VCard,
        VCardText,
        VCardTitle
    },
    mounted () {

    }
  }
</script>