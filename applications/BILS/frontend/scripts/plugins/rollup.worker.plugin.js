import {basename, extname} from 'path'
import {cleanUrl, parseRequest} from './util'

/**
 * Resolve ?worker import to the function creating the worker object
 * @returns {import('rollup').Plugin}
 */
export default function createWorkerPlugin() {
  return {
    name: 'electron:worker',

    resolveId(id, importer) {
      const query = parseRequest(id)
      if (typeof query.worker === 'string') {
        return id + `&importer=${importer}`
      }
    },
    load(id) {
      const {worker, importer} = parseRequest(id)
      if (typeof worker === 'string' && typeof importer === 'string') {
        // emit as separate chunk
        const cleanPath = cleanUrl(id)
        const ext = extname(cleanPath)
        const hash = this.emitFile({
          type: 'chunk',
          id: cleanPath,
          fileName: `${basename(cleanPath, ext)}.worker.js`,
          importer: importer
        })
        const path = `__ASSETS__${hash}__`
        if (this.meta.watchMode) {
          return `
          import { Worker } from 'worker_threads';
          export default function (options) { return new Worker(${path}, options); }`
        } else {
          return `
          import { join } from 'path';
          import { Worker } from 'worker_threads';
          export default function (options) { return new Worker(join(__dirname, ${path}), options); }`
        }
      }
    }
  }
}
