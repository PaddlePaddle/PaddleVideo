import { ffModel } from './model/FFModel'
import { JsonOutput } from './output/JsonOutput'
import BaseModel from './model/BaseModel'
import BaseOutput from './output/BaseOutput'
import { useService, useShell } from '/@/composables'
const { writFile } = useService('FileService')
interface exportParams {
  path?: string | null,
  modelType: string,
  outputType: string,
  filterArr?: any | null,
  data: any,
  // eslint-disable-next-line no-undef
  encoding?: BufferEncoding | null
}

export async function exportModel (params: exportParams) {
  const outputData = await getExportModel(params)
  await writFile(params.path as string, outputData as string, params.encoding || 'utf-8')
  // await shell.openPath(await getParent(params.path as string))
}

export async function getExportModel (params: exportParams) {
  const handler = getHandler(params.modelType, params.outputType)
  if (handler.model == null) {
    return
  }
  if (handler.output == null) {
    return
  }
  const modelData = await handler.model.genModel(params)
  const outputData = await handler.output.output(modelData)
  return outputData
}

function getHandler (modelType: string, outputType: string) : { model: BaseModel | null, output: BaseOutput | null } {
  let model = null
  let output = null
  if (modelType === 'FF') {
    model = ffModel
  }
  if (outputType === 'json') {
    output = JsonOutput
  }
  return {
    model,
    output
  }
}

export {
  exportParams
}
