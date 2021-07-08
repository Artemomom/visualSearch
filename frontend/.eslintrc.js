const path = require('path');
module.exports = {
  parser: 'babel-eslint',
  extends: [
    'airbnb',
    'plugin:import/errors',
    'plugin:import/warnings',
    'prettier',
    'prettier/react',
    'plugin:react-hooks/recommended'
  ],
  plugins: ['flowtype', 'prettier', 'import'],
  rules: {
    'prefer-destructuring': ['warn', { object: true, array: false }],
    'linebreak-style': 0,
    'prefer-const': 0,
    'spaced-comment': 0,
    'padded-blocks': 0,
    'import/imports-first': 0,
    'import/prefer-default-export': 0,
    'comma-dangle': 0,
    'no-shadow': 0,
    'arrow-body-style': 0,
    'quote-props': 0,
    'no-unused-vars': 1,
    'consistent-return': 0,
    'max-len': 0,
    'no-use-before-define': ['error', { functions: false, classes: true }],
    'no-underscore-dangle': 'off',
    'jsx-a11y/href-no-hash': 'off',
    'jsx-a11y/anchor-is-valid': ['warn', { aspects: ['invalidHref'] }],
    'jsx-a11y/label-has-associated-control': 'off',
    'jsx-a11y/click-events-have-key-events': 'off',
    'jsx-a11y/no-static-element-interactions': 'off',
    'react/prefer-stateless-function': ['off'],
    'react/state-in-constructor': 'off',
    'react/jsx-filename-extension': ['off'],
    'react/forbid-prop-types': ['off'],
    'react/require-default-props': 'off',
    'react/jsx-props-no-spreading': 'off',
    'react/jsx-curly-brace-presence': 'off',
    'react/no-danger': 0,
    'react/prop-types': 0,
    'no-plusplus': 'off',
    'no-throw-literal': 'error',
    'no-bitwise': ['off'],
    'dot-notation': ['off'],
    'prettier/prettier': ['warn'],
    'flowtype/define-flow-type': 1,
    camelcase: ['off'],
    'jsx-a11y/label-has-for': ['off'],
    'no-nested-ternary': 'off',
    'react/no-this-in-sfc': 'off',
    'no-param-reassign': 'off',
    'react/sort-comp': [
      1,
      {
        order: [
          'static-methods',
          'instance-variables',
          'lifecycle',
          '/^on.+$/',
          'everything-else',
          'rendering'
        ],
        groups: {
          rendering: ['/^render.+$/', 'render']
        }
      }
    ]
  },
  parserOptions: {
    ecmaVersion: 6,
    sourceType: 'module',
    ecmaFeatures: {
      spread: true,
      legacyDecorators: true
    }
  },
  globals: {
    global: true,
    document: true,
    window: true
  },
  settings: {
    'import/extensions': ['.js', '.jsx'],
    'import/parser': 'babel-eslint',
    'import/resolver': {
      alias: [
        ['src', path.join(__dirname, 'src/')],
        ['public', path.join(__dirname, 'public/')]
      ]
    }
  }
};
