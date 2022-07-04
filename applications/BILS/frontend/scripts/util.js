const {join} = require('path')
const loadConfigFile = require('rollup/dist/loadConfigFile')

/**
 * Load rollup config
 * @returns {Promise<import('rollup').RollupOptions[]>}
 */
async function loadRollupConfig() {
  const {options, warnings} = await loadConfigFile(join(__dirname, 'rollup.config.js'), {})

  warnings.flush()

  return options
}

module.exports = {
  loadRollupConfig
}
