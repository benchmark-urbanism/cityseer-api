module.exports = {
  root: true,
  env: {
    node: true,
  },
  extends: ['plugin:prettier/recommended', 'plugin:vue/vue3-recommended'],
  plugins: ['simple-import-sort'],
  rules: {
    'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'vue/multi-word-component-names': 'off',
    'simple-import-sort/imports': 'error',
  },
}
