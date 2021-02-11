<template>
  <div class="container">
    <div class="column">
      <div class="elem-name">
        {{ name }}
      </div>
      <div class="elem-type">
        {{ type }}
      </div>
    </div>
    <div class="elem-content">
      <slot></slot>
    </div>
  </div>
</template>

<style lang="postcss" scoped>

.container {
  display: flex;
  flex: 1;
  margin-top: 30px;
  margin-bottom: 30px;
}

.column {
  display: flex;
  flex-direction: column;
  min-width: 200px;
  width: 30%;
}

.elem-content {
  flex-grow: 1;
  width: 70%;
}

.elem-name {
  font-size: 16px;
  font-family: monospace;
  font-weight: bold;
}

.elem-type {
  font-size: 12px;
}

p {
  margin-top: 0;
}

@media screen and (max-width: 680px) {
  .container {

    display: block;
  }

  .elem-content {

    padding-top: 5px;
    width: 100%;
  }
}

</style>

<script>

export default {
  name: 'DisplayFunction',
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
