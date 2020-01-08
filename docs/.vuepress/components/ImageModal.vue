<template>
  <div @click="dialogActive=!dialogActive">
    <img
            :src="path"
            :alt="alt"
    >
    <em> {{ caption }} </em>
    <div
            v-if="dialogActive"
            class="modal"
    >
      <div class="modal-content">
        <img
                :src="path"
                :alt="'boo'"
        >
        <em> {{ caption }} </em>
      </div>
    </div>
  </div>
</template>

<style lang="stylus">

  .modal
    position: fixed
    z-index: 100 !important
    left: 0
    top: 0
    width: 100%
    height: 100%
    overflow: auto
    background-color: rgba(0, 0, 0, 0.6)

  /* Modal Content/Box */
  .modal-content
    position: relative
    background-color: #2e2e2e
    margin: 15% auto
    padding: 20px
    width: 80%
    max-width: 1500px

</style>

<script>
  export default {
    name: 'ImageModal',
    props: {
      path: {
        type: String,
        required: true
      },
      alt: {
        type: String,
        required: false,
        default: ''
      },
      caption: {
        type: String,
        required: false,
        default: ''
      }
    },
    data () {
      return {
        dialogActive: false
      }
    }
  }
</script>
