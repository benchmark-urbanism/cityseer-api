module.exports = {
  printWidth: 100,
  semi: false,
  singleQuote: true,
  pugAttributeSeparator: 'none',
  pugCommentPreserveSpaces: 'trim-all',
  pugWrapAttributesThreshold: -1,
  pugEmptyAttributes: 'as-is',
  plugins: [require('prettier-plugin-tailwindcss'), require('prettier-plugin-astro')],
  tailwindConfig: './tailwind.config.cjs',
  overrides: [
    {
      files: '*.astro',
      options: {
        parser: 'astro',
      },
    },
  ],
}
