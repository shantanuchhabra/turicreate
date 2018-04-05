var path = require('path');
const APP_DIR = path.resolve(__dirname, 'lib');

module.exports = {
  entry: './node_modules/vega/bin/vg2png',
  target: 'node',
  module: {
    rules: [{
      test: /\.node$/,
      use: 'node-loader'
    }]
  },
  output: {
    path: path.join(__dirname, 'build'),
    filename: 'vg2png.js'
  }
}
