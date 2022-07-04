import { useProject, Project } from '/@/store/project'
import { PACKAGE_FOLDER, TIMELINE_THUMBNAIL_FOLDER } from '/@/constants'
import { useService } from '/@/composables'
import { useLabelDefineStore, useLabelsStore } from '/@/store'
import { exportModel } from '/@/handler/export'
import { ElMessage } from 'element-plus/es'

const { ensureFile } = useService('FileService')
const fileService = useService('FileService')
interface ProjectExportConfig {
  projectConfig: Project,
  projectTimelineImgFileMap: any,
  projectLabelDefine: any,
  projectLabels: any
}
export async function exportConfigFile(path: string) {
  const projectStore = useProject()
  const labelStore = useLabelsStore()
  const labelDefine = useLabelDefineStore()
  const projectConfig = projectStore.project
  console.log(projectConfig)
  const timelineImgFileMap = await getTimelineImgFileMap()
  const projectExportConfig : ProjectExportConfig = {
    projectConfig: projectConfig,
    projectTimelineImgFileMap: timelineImgFileMap,
    projectLabelDefine: labelDefine.labeldefine,
    projectLabels: labelStore.getLabels()
  }
  await fileService.writFile(path, JSON.stringify(projectExportConfig), 'utf-8')
}

async function getTimelineImgFileMap() {
  const fileMap: any = {}
  const projectStore = useProject()
  const projectConfig = projectStore.project
  const parentFolderPath =
    projectStore.baseConfig.projectPath +
    (await fileService.getSep()) +
    PACKAGE_FOLDER
  for (const videoListElement of projectConfig.videoList) {
    const folderPath =
      parentFolderPath +
      (await fileService.getSep()) +
      TIMELINE_THUMBNAIL_FOLDER +
      (await fileService.getSep()) +
      videoListElement.uuid
    if (!videoListElement.isGenTimeLineThumbnail) {
      continue
    }
    try {
      fileMap[videoListElement.uuid as string] = []
      const filePathList = await fileService.getChildPaths(folderPath)
      for (const filePath of filePathList) {
        const imgPath = folderPath + (await fileService.getSep()) + filePath
        const img = await fileService.readFile(imgPath, 'binary')
        const imgFileName = await fileService.getFileName(imgPath)
        fileMap[videoListElement.uuid as string].push({
          fileName: imgFileName,
          imgBinary: img
        })
      }
    } catch (err) {
      console.error(err)
    }
  }
  return fileMap
}

export async function importConfigFile(path: string) {
  const projectStore = useProject()
  const labelStore = useLabelsStore()
  const labelDefine = useLabelDefineStore()
  if (!await fileService.isFile(path)) {
    throw new Error('路径不存在')
  }
  const readData = await fileService.readFile(path, 'utf-8')
  const dataJson = JSON.parse(readData as string)
  console.log(dataJson)
  projectStore.importProjectConfig(dataJson.projectConfig)
  await importTimelineImgFileMap(dataJson)
  labelDefine.setLabelDefine(dataJson.projectLabelDefine)
  labelStore.setProjectLabels(dataJson.projectLabels)
}

export async function currentExportJson (path: string) {
  await ensureFile(path)
  const projectStore = useProject()
  const Label = useLabelsStore()
  const labels = Label.getAllLabels()
  const treeArray = labels
  await exportModel({
    path: path as string,
    modelType: 'FF',
    outputType: 'json',
    data: treeArray,
    encoding: null
  })
  if (!projectStore.baseConfig.outNoTip) {
    ElMessage({
      message: '导出模型成功',
      type: 'success'
    })
  }
}

async function importTimelineImgFileMap (projectExportConfig: ProjectExportConfig) {
  const projectStore = useProject()
  const parentFolderPath =
    projectStore.baseConfig.projectPath +
    (await fileService.getSep()) +
    PACKAGE_FOLDER
  for (const uuid in projectExportConfig.projectTimelineImgFileMap) {
    const folderPath =
      parentFolderPath +
      (await fileService.getSep()) +
      TIMELINE_THUMBNAIL_FOLDER +
      (await fileService.getSep()) +
      uuid
    const imgArr = projectExportConfig.projectTimelineImgFileMap[uuid]
    for (let i = 0; i < imgArr.length; i++) {
      const item = imgArr[i]
      await fileService.writFile(folderPath + await fileService.getSep() + item.fileName, item.imgBinary, 'binary')
    }
  }
}
