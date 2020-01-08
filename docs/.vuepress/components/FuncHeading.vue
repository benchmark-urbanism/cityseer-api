<template>
  <div class="heading-style">
    <slot></slot>
  </div>
</template>

<style lang="stylus" scoped>

  .heading-style
    padding 1em 0
    margin 10px 0
    font-size 14px
    font-weight 500
    border-bottom-style solid
    border-color $borderColor
    border-width 1px
</style>
