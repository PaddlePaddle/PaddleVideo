import {basename, extname, join} from 'path'
import {cleanUrl} from './util'

/**
 * Resolve the import of preload and emit it as single chunk of js file in rollup.
 * @returns {import('rollup').Plugin}
 */
export default function createPreloadPlugin() {
  return {
    name: 'electron:preload',

    resolveId(source) {
      if (source.startsWith('/@preload')) {
        return source.replace('/@preload', join(__dirname, '..', 'src', 'preload')) + '?preload'
      }
    },
    async load(id) {
      if (id.endsWith('?preload')) {
        const clean = cleanUrl(id)
        const ext = extname(clean)
        const hash = this.emitFile({
          type: 'chunk',
          id: clean,
          fileName: `${basename(cleanUrl(id), ext)}.preload.js`
        })
        const path = `__ASSETS__${hash}__`
        if (this.meta.watchMode) {
          return `export default ${path}`
        } else {
          return `import { join } from 'path'; export default join(__dirname, ${path})`
        }
      }
    }
  }
}
