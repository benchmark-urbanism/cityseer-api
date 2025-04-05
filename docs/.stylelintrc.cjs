module.exports = {
  extends: ["stylelint-config-recommended", "stylelint-config-tailwindcss"],
  plugins: [],
  rules: {
    'selector-nested-pattern': '^&',
  },
}
