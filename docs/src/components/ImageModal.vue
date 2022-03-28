<template lang="pug">
div
  img(:src='path' :alt='alt')
  em {{ caption }}
</template>

<script>
export default {
  name: 'ImageModal',
  props: {
    path: {
      type: String,
      required: true,
    },
    alt: {
      type: String,
      required: false,
      default: '',
    },
    caption: {
      type: String,
      required: false,
      default: '',
    },
  },
}
</script>

<style lang="postcss" scoped>
.modal {
  @apply fixed w-full h-full z-50;

  background-color: rgba(0, 0, 0, 0.9);
}

/* Modal Content/Box */
.modal-content {
  @apply relative bg-dark-grey p-4;

  margin: 15% auto;
  width: 80%;
  max-width: 1500px;
}
</style>
