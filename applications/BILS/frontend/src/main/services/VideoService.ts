import { Service } from '/@main/services/Service'
import { VideoItem } from '../../renderer/store/project'
import { FileService } from './FileService'
import { THUMBNAIL_FOLDER, TIMELINE_THUMBNAIL_FOLDER, PACKAGE_FOLDER } from '../../renderer/constants/videoConstants'

const ffmpegPath = require('@ffmpeg-installer/ffmpeg')
const ffprobePath = require('@ffprobe-installer/ffprobe')
const ffmpeg = require('fluent-ffmpeg')
if (process.env.NODE_ENV !== 'development') {
  ffmpeg.setFfmpegPath(ffmpegPath.path.replace('app.asar', 'app.asar.unpacked'))
  ffmpeg.setFfprobePath(ffprobePath.path.replace('app.asar', 'app.asar.unpacked'))
} else {
  ffmpeg.setFfmpegPath(ffmpegPath.path)
  ffmpeg.setFfprobePath(ffprobePath.path)
}
const fileService = new FileService(null)

export class VideoService extends Service {
  async genThumbnail(projectPath: string, videoItem: VideoItem): Promise<string> {
    const parentFolderPath = projectPath + await fileService.getSep() + PACKAGE_FOLDER
    const folderPath = parentFolderPath + await fileService.getSep() + THUMBNAIL_FOLDER
    const fileName = videoItem.uuid + '-thumbnail.png'
    await fileService.ensureFolder(folderPath)
    await fileService.hideFile(parentFolderPath)
    await _genThumbnail(videoItem.path, ['20%'], folderPath, fileName)
    return await fileService.getFilePrefix() + folderPath + await fileService.getSep() + fileName
  }

  async getVideoMeta (projectPath: string, videoItem: VideoItem) {
    const videoMeta:any = await _getVideoMeta(videoItem)
    const duration = Math.floor(videoMeta.format.duration * 100) / 100
    const videoName = videoMeta.format.filename.split('/').pop()
    const Meta = {
      duration: duration,
      fps: videoMeta.streams[0].r_frame_rate,
      size: (videoMeta.format.size / 1e6).toFixed(2) + 'MB',
      resolution: videoMeta.streams[0].width + '×' + videoMeta.streams[0].height,
      videoName: videoName,
      uuid: ''
    }
    return Meta
  }

  async getTimeLineThumbnail (projectPath: string, videoItem: VideoItem) {
    const videoMeta:any = await _getVideoMeta(videoItem)
    const duration = Math.floor(videoMeta.format.duration * 100) / 100
    const timestamps:Array<number | string> = []
    const parentFolderPath = projectPath + await fileService.getSep() + PACKAGE_FOLDER
    const folderPath = parentFolderPath + await fileService.getSep() + TIMELINE_THUMBNAIL_FOLDER + await fileService.getSep() + videoItem.uuid
    const fileName = '%i.png'
    await fileService.ensureFolder(folderPath)
    await fileService.hideFile(parentFolderPath)
    const space = Math.floor(duration / 30)
    let timeSpace = space
    timestamps.push('0%')
    for (let i = 0; i < 28; i++) {
      timestamps.push(timeSpace)
      timeSpace += space
    }
    timestamps.push('99%')
    console.log('开始生成')
    await _genThumbnail(videoItem.path, timestamps, folderPath, fileName)
    console.log('生成结束')
    return duration
  }
}

/**
 * 生成缩略图
 * @param projectPath
 * @param videoItem
 * @param sep
 */
function _genThumbnail (videoPath: string | null, timestamps: Array<string | number>, folderPath: string, fileName: string) {
  var number = 0
  return new Promise((resolve) => {
    ffmpeg(videoPath as string).screenshots({
      timestamps: timestamps,
      filename: fileName,
      folder: folderPath,
      size: '320x240'
    }).on('end', function () {
      resolve(null)
    }).on('error', function (err: any) {
      console.log(err)
      resolve(null)
    }).on('progress', function (progress) {
      number++
      console.log(JSON.stringify(progress))
    })
  })
}

/**
 * 获取视频信息
 * @param path
 */
function _getVideoMeta(videoItem: VideoItem) {
  return new Promise((resolve, reject) => {
    ffmpeg(videoItem.path).ffprobe((err: any, data: any) => {
      if (err) {
        reject(err)
      }
      resolve(data)
    })
  })
}
