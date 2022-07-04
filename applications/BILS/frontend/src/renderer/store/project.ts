import { defineStore } from 'pinia'
import { useStorage } from '@vueuse/core'
import { useService } from '/@/composables'
import { bus } from '/@/utils/busUtils'
import { PLAY_VIDEO } from '/@/constants'

interface BaseConfig {
  projectName: string | null
  projectPath: string | null
  fps: number | null
  outNoTip: boolean | null
  autoSave: boolean | null
  outContent: string | null
  dataPath: string | null
  format: string | null
  desc: string | null
}

interface VideoItem {
  uuid: string | null,
  path: string | null,
  thumbnail: string | null,
  suffix: string | null,
  name: string | null,
  isLoadTimeLineThumbnail?: boolean | null,
  isLoadTimeLineThumbnailFinish ?: boolean | null,
  isPlayed: boolean | null,
  isLabeled: boolean | null,
  isGenTimeLineThumbnail?: boolean | null,
  duration?: number | null
}

enum FileType {
  FOLDER,
  File
}

interface FileItem {
  uuid: string | null,
  path: string | null,
  name: string | null,
  type: FileType | null,
  suffix: string | null,
  isLoadChildren: Boolean | null,
  children: Array<FileItem>
}

interface Project {
  baseConfig: BaseConfig
  videoList: Array<VideoItem>
  videoMeta: Array<Object>
  curVideo: string | null
  fileList: Array<FileItem>
  state: Object,
  exportConfigArr: Array<any>,
  isPanelShow: Boolean,
  isBrowserShow: Boolean
}

const _project: Project = {
  baseConfig: {
    projectName: '',
    projectPath: '',
    fps: 25,
    outNoTip: false,
    autoSave: false,
    outContent: 'all',
    dataPath: '',
    format: 'json',
    desc: ''
  },
  videoList: [],
  videoMeta: [],
  curVideo: '',
  fileList: [],
  exportConfigArr: [],
  isPanelShow: true,
  isBrowserShow: true
}
const { getTimeLineThumbnail, getVideoMeta } = useService('VideoService')
const useProject = defineStore({
  id: 'useProject',
  state: () => {
    const project = useStorage('project', _project)
    project.value.videoList.forEach((item:any) => {
      item.isLoadTimeLineThumbnail = false
    })
    return {
      project: project
    }
  },
  getters: {
    baseConfig (state) {
      return state.project.baseConfig
    },
    videoList (state) {
      return state.project.videoList
    },
    fileList (state) {
      return state.project.fileList
    },
    isBrowserShow (state) {
      return state.project.isBrowserShow
    },
    isPanelShow (state) {
      return state.project.isPanelShow
    },
    exportConfigArr (state) {
      return state.project.exportConfigArr == null ? [] : state.project.exportConfigArr
    },
    curVideo (state) {
      return state.project.curVideo
    },
    curVideoItem (state) {
      const nowVideoIndex = state.project.videoList.findIndex(item => {
        return item.uuid === state.project.curVideo
      })
      if (nowVideoIndex >= 0) {
        return state.project.videoList[nowVideoIndex]
      }
      return null
    }
  },
  actions: {
    videoMeta (uuid: string) {
      try {
        const videoItemIndex = this.project.videoMeta.findIndex((item: any) => {
          return item.uuid === uuid
        })
        return this.project.videoMeta[videoItemIndex]
      } catch (err) {
        return {}
      }
    },
    async loadVideo(uuid: string, next: number|null) {
      let videoItemIndex = this.project.videoList.findIndex((item: any) => {
        return item.uuid === uuid
      })
      if (next === 1)videoItemIndex = Math.min(videoItemIndex + 1, this.project.videoList.length - 1)
      else if (next === -1)videoItemIndex = Math.max(videoItemIndex - 1, 0)
      const videoItem = this.project.videoList[videoItemIndex]
      this.project.curVideo = videoItem.uuid
      if (videoItem.isLoadTimeLineThumbnail) {
        return
      }
      videoItem.isLoadTimeLineThumbnail = true
      if (!videoItem.isGenTimeLineThumbnail) {
        this.loadTimeLineThumbnail(videoItem)
      }
      videoItem.isLoadTimeLineThumbnail = false
      videoItem.isPlayed = true
      bus.emit(PLAY_VIDEO, videoItem)
    },

    async loadTimeLineThumbnail (videoItem) {
      const duration = await getTimeLineThumbnail(
        this.project.baseConfig.projectPath as string,
        videoItem
      )
      videoItem.isGenTimeLineThumbnail = true
      videoItem.duration = duration
      videoItem.isLoadTimeLineThumbnailFinish = true
    },

    changeProjectBaseConfig (baseConfig: BaseConfig) {
      this.project.baseConfig = baseConfig
    },
    resetVideo () {
      this.project.videoList = []
    },
    resetFileList () {
      this.project.fileList = []
    },
    async addVideo (video: VideoItem) {
      const index = this.project.videoList.findIndex((item: VideoItem) => {
        return item.name === video.name
      })
      if (index >= 0) {
        return
      }
      video.isLoadTimeLineThumbnail = video.isLoadTimeLineThumbnail == null ? false : video.isLoadTimeLineThumbnail
      video.isGenTimeLineThumbnail = video.isGenTimeLineThumbnail == null ? false : video.isGenTimeLineThumbnail
      this.project.videoList.push(video)
      const Meta = await getVideoMeta(
        this.project.baseConfig.projectPath as string,
        video
      )
      Meta.uuid = video.uuid
      this.project.videoMeta.push(Meta)
      // console.log('Meta----------------------------->', Meta)
    },
    setFileItem (fileItem : FileItem) {
      this.project.fileList.push(fileItem)
    },
    importProjectConfig (projectConfig: Project) {
      for (const key in projectConfig) {
        // @ts-ignore
        this.project[key] = projectConfig[key]
      }
    },
    changeIsBrowserShow (flag: Boolean) {
      this.project.isBrowserShow = flag
    },
    changeIsPanelShow (flag: Boolean) {
      this.project.isPanelShow = flag
    },
    changeExportConfigArr (arr: Array<any>) {
      this.project.exportConfigArr = arr
    },
    resetCurVideo () {
      this.project.curVideo = ''
    },
    resetVideoMeta () {
      this.project.videoMeta = []
    }
  }
})

export {
  Project,
  useProject,
  BaseConfig,
  VideoItem,
  FileType,
  FileItem
}
