<template>
    <div class="chip-style" :class=" { important: important }">
        {{ text }}
    </div>
</template>

<style lang="stylus">

    .chip-style
        display inline-block
        font-size 12px
        height 20px
        line-height 20px
        border-radius 2px
        padding 0 6px
        color white
        margin-right 5px
        background-color $info
        vertical-align top
    &.important
        background-color $secondary

</style>

<script>
  export default {
    name: 'Chip',
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
    }
  }
</script>