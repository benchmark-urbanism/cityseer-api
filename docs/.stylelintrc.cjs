module.exports = {
  extends: [
    'stylelint-config-standard',
    // last - to turn of clashes with prettier
    'stylelint-config-prettier',
  ],
  plugins: [],
  rules: {
    'selector-nested-pattern': '^&',
  },
}
