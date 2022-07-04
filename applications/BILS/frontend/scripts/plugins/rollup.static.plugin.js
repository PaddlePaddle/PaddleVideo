import {existsSync, readFile} from 'fs-extra'
import {basename, join} from 'path'
import {cleanUrl} from './util'

/**
 * Resolve import of static resource to real static resource path
 * @returns {import('rollup').Plugin}
 */
export default function createStaticPlugin() {
  return {
    name: 'electron:static',

    resolveId(source) {
      if (source.startsWith('/@static')) {
        const target = source.replace('/@static', join(__dirname, '../static'))
        if (existsSync(target)) {
          return target + '?static'
        }
      }
    },
    async load(id) {
      if (id.endsWith('?static')) {
        const clean = cleanUrl(id)
        if (this.meta.watchMode) {
          // dev mode just return the file path
          return `export default ${JSON.stringify(clean)}`
        } else {
          const hash = this.emitFile({
            fileName: join('static', basename(clean)),
            type: 'asset',
            source: await readFile(clean)
          })
          return `import { join } from 'path'; export default join(__dirname, __ASSETS__${hash}__);`
        }
      }
    }
  }
}
