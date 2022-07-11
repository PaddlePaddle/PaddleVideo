import BaseOutput from '/@/handler/export/output/BaseOutput'
import { useService } from '/@/composables'
const { format } = useService('PrettierService')
const config = {
  trailingComma: 'es5',
  tabWidth: 4,
  semi: false,
  singleQuote: true,
  parser: 'json'
}
export const JsonOutput: BaseOutput = {
  async output (model) {
    return format(JSON.stringify(model), config)
  }
}
