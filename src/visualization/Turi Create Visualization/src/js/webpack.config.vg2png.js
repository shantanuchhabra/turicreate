var path = require('path');
const APP_DIR = path.resolve(__dirname, 'lib');

module.exports = {
  entry: './node_modules/vega/bin/vg2png',
  target: 'node',
  module: {
        rules: [
            {
                loader: 'babel-loader',
                // test: [/.js$/, 'vg2png'],
                // include : [APP_DIR, './node_modules/vega/bin/'],
                options: {
                    presets: [ 'es2015', 'react' ]
                }
            }
        ]
    },
  output: {
    path: path.join(__dirname, 'build'),
    filename: 'vg2png.js'
  }
}