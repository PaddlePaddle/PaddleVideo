import MagicString from 'magic-string'
import {join} from 'path'

export const assetsUrlRE = /__ASSETS__([a-z\d]{8})__(?:\$_(.*?)__)?/g

/**
 * Replace internal assets symbol to real asset file path
 * @type {() => import('rollup').Plugin}
 */
const createPlugin = () => ({
  name: 'electron:assets',
  renderChunk(code, chunk) {
    /**
     * @type {MagicString | undefined}
     */
    let s
    for (let result = assetsUrlRE.exec(code); result; result = assetsUrlRE.exec(code)) {
      s = s || (s = new MagicString(code))
      const [full, hash] = result
      const fileName = this.getFileName(hash)
      if (this.meta.watchMode) {
        s.overwrite(result.index, result.index + full.length, JSON.stringify(join(__dirname, '../dist', fileName)))
      } else {
        s.overwrite(result.index, result.index + full.length, JSON.stringify(fileName))
      }
    }
    if (s) {
      return {code: s.toString(), map: s.generateMap({hires: true})}
    } else {
      return null
    }
  }
})

export default createPlugin
