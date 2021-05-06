module.exports = {
  // override vscode settings
  // Notice: It only affects the settings used by Vetur.
  settings: {},
  // **optional** default: `[{ root: './' }]`
  // support monorepos
  projects: [
    "./docs/", // shorthand for only root.
    {
      // **required**
      // Where is your project?
      // It is relative to `vetur.config.js`.
      root: "./docs/",
      // **optional** default: `'package.json'`
      // Where is `package.json` in the project?
      // We use it to determine the version of vue.
      // It is relative to root property.
      package: "./docs/package.json",
    },
  ],
};
