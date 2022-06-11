/**
 * @type {import('electron-builder').Configuration}
 */
const config = {
  productName: '',
  appId: '',
  electronVersion: process.env.ELECTRON_VERSION, // only used for development debugging
  directories: {
    output: 'build',
    buildResources: 'build',
    app: 'dist'
  },
  // assign publish for auto-updater
  // set this to your own repo!
  // publish: [{
  //   provider: 'github',
  //   owner: '',
  //   repo: ''
  // }],
  files: [
    // don't include node_modules as all js gallery are bundled into production js by rollup
    // unless you want to prevent some module to bundle up
    // list them below
  ]
}

module.exports = config
