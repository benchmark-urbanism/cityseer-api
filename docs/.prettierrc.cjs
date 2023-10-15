module.exports = {
  printWidth: 100,
  semi: false,
  singleQuote: true,
  pugAttributeSeparator: 'as-needed',
  pugCommentPreserveSpaces: 'trim-all',
  pugSortAttributes: 'asc',
  pugWrapAttributesThreshold: 1,
  pugFramework: 'vue',
  pugEmptyAttributes: 'all',
  pugClassLocation: 'after-attributes',
  pugExplicitDiv: true,
  plugins: ['prettier-plugin-astro'],
  overrides: [
    {
      files: '*.astro',
      options: {
        parser: 'astro',
      },
    },
  ],
}
