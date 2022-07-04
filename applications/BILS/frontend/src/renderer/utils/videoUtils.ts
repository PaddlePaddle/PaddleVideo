import { useService } from '/@/composables'
import { useProject, VideoItem, BaseConfig } from '/@/store/project'
import { v1 as uuidv4 } from 'uuid'
import { VIDEO_SUFFIX_LIST } from '/@/constants/videoConstants'
const { getAllFile, getFileName, getFileExpandName } = useService('FileService')
const { genThumbnail } = useService('VideoService')

export async function loadAllVideo (config: BaseConfig) {
  if (config.dataPath == null) {
    return
  }
  if (config.projectPath == null) {
    return
  }
  const projectStore = useProject()
  projectStore.resetVideo()
  const filePathList = await getAllFile(config.dataPath, VIDEO_SUFFIX_LIST)
  let firstVideoUuid = null
  for (let i = 0; i < filePathList.length; i++) {
    const path = filePathList[i]
    const videoItem: VideoItem = {
      uuid: uuidv4(),
      name: await getFileName(path),
      path: path,
      suffix: await getFileExpandName(path),
      isPlayed: false,
      isLabeled: false,
      isLoadTimeLineThumbnailFinish: false,
      thumbnail: null
    }
    const thumbnail = await genThumbnail(config.projectPath, videoItem)
    videoItem.thumbnail = thumbnail
    projectStore.addVideo(videoItem)
    if (i === 0) {
      firstVideoUuid = videoItem.uuid
    }
  }
  if (filePathList.length > 0) {
    await projectStore.loadVideo(firstVideoUuid as string, null)
  }
}
