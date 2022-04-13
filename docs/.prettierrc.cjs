module.exports = {
  printWidth: 100,
  semi: false,
  singleQuote: true,
  pugAttributeSeparator: 'none',
  pugCommentPreserveSpaces: 'trim-all',
  pugWrapAttributesThreshold: -1,
  pugEmptyAttributes: 'as-is',
  plugins: [require('prettier-plugin-tailwindcss')],
  tailwindConfig: './tailwind.config.cjs',
}
