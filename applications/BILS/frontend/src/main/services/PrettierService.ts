import { Service } from './Service'
const prettier = require('prettier')
export class PrettierService extends Service {
  /**
   * Example for inject and shared lib
   */
  async format(content: string, config: Object) {
    return prettier.format(content, config)
  }
}
