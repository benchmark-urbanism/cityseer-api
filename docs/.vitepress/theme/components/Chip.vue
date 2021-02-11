<template>
  <div
          :class="{ important }"
          class="chip-style"
  >
    {{ text }}
  </div>
</template>

<style lang="postcss" scoped>

.chip-style {
  display: inline-block;
  font-size: 12px;
  height: 20px;
  line-height: 20px;
  border-radius: 2px;
  padding: 0 6px;
  color: white;
  margin-right: 5px;
  background-color: #d32f2f;
  vertical-align: top;
}
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
