import {join} from 'path'

/**
 * Provide vue-devtool extension virtual import.
 * @returns {import('rollup').Plugin}
 */
export default function createVueDevtoolsPlugin() {
  return {
    name: 'electron:devtools',
    async resolveId(id) {
      if (id === 'vue-devtools') {
        return id
      }
    },
    async load(id) {
      if (id === 'vue-devtools') {
        const path = join(__dirname, '../extensions')
        return `export default ${JSON.stringify(path)}`
      }
    }
  }
}
