<template>
  <div class="sig-style">
    <slot></slot>
  </div>
</template>

<style lang="postcss" scoped>

.sig-style {
  margin: 30px 0;
  padding: 1.1em;
  font-size: 18px;
  font-family: monospace;
  font-weight: bold;
  color: #ffffff;
  background-color: #2e2e2e;
  border-radius: 10px;
}

/* necessary for when wrapping signatures with <pre></pre> tags for spacing */
.sig-style pre {
  margin: 0;
  padding: 0;
}
</style>
