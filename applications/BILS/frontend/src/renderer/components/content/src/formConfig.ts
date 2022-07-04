import { reactive } from 'vue'

import { useProject, BaseConfig } from '/@/store/project'
import { useService } from '/@/composables'
const { isFolder } = useService('FileService')
const checkIsFolder = async (rule: any, value: any, callback: any) => {
  try {
    await isFolder(value)
    callback()
  } catch (err) {
    callback(new Error('请输入正确的文件夹路径'))
  }
}

export const form = reactive<BaseConfig>({
  projectName: null,
  projectPath: null,
  fps: null,
  outNoTip: null,
  autoSave: null,
  outContent: null,
  dataPath: null,
  format: null,
  desc: null
})

export const rules = reactive({
  projectName: [
    {
      required: true,
      type: 'string',
      message: '项目名称不能为空'
    }
  ],
  projectPath: [
    {
      required: true,
      type: 'string',
      message: '项目目录不能为空'
    },
    {
      validator: checkIsFolder
    }
  ],
  dataPath: [
    {
      required: true,
      type: 'string',
      message: '数据集目录不能为空'
    },
    {
      validator: checkIsFolder
    }
  ]
})

export default function getFormConfig () {
  const projectStore = useProject()
  console.log(projectStore.baseConfig)
  for (const key in projectStore.baseConfig) {
    // @ts-ignore
    form[key] = projectStore.baseConfig[key]
  }
  return {
    form,
    rules
  }
}
