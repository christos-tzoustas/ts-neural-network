module.exports = {
  env: {
    node: true,
    jest: true,
    browser: true,
  },
  extends: ["plugin:@typescript-eslint/recommended", "eslint:recommended"],
  parser: "@typescript-eslint/parser",
  plugins: ["@typescript-eslint"],
  rules: {
    "no-unused-vars": ["error", { args: "none" }],
  },
  root: true,
};
