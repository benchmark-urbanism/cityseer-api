/* eslint-env node */
require('@rushstack/eslint-patch/modern-module-resolution')

module.exports = {
  root: true,
  env: {
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:vue/vue3-recommended', // eslint-plugin-vue
    'plugin:astro/recommended',
    // last - to turn of clashes with prettier
    'prettier', // eslint-config-prettier
  ],
  // for parsing vue <template>
  parser: 'vue-eslint-parser',
  // for parsing <script>
  parserOptions: {
    parser: '@typescript-eslint/parser',
    sourceType: 'module',
  },
  // sorting imports and tailwind plugin
  plugins: ['simple-import-sort', 'tailwindcss', '@typescript-eslint'],
  // rules for enabling import sorting errors
  rules: {
    'simple-import-sort/imports': 'error',
    'simple-import-sort/exports': 'error',
  },
  overrides: [
    {
      files: ['*.astro'],
      parser: 'astro-eslint-parser',
      parserOptions: {
        parser: '@typescript-eslint/parser',
        extraFileExtensions: ['.astro'],
      },
      rules: {},
    },
  ],
}
