import { Service } from '/@main/services/Service'
import { sep } from 'path'
import fs from 'fs/promises'
import * as originFs from 'fs'
import fsExtraPromise from 'fs-extra-promise'
const exec = require('child_process').exec
export class FileService extends Service {
  /**
   * 获取分割符号
   */
  async getSep () {
    return sep
  }

  /**
   * 根据路径获取文件名称
   * @param path
   */
  async getFileName(path: string) : Promise<string> {
    path = await this.getGeneralPath(path)
    const lastIndex = path.lastIndexOf(sep)
    const name = path.substring(lastIndex + 1)
    return name
  }

  /**
   * 根据路径获取文件扩展名
   * @param path
   */
  async getFileExpandName (path: string) {
    const name = await this.getFileName(path)
    const lastIndex = name.lastIndexOf('.')
    const expandName = name.substring(lastIndex + 1)
    return expandName.toLowerCase()
  }

  /**
   * 判断是否是文件夹
   * @param path
   */
  async isFolder(path: string) : Promise<boolean> {
    try {
      path = await this.getGeneralPath(path)
      const file = await fs.lstat(path)
      return file.isDirectory()
    } catch (err) {
      console.error('[' + path + ']路径不存在', err)
      throw new Error('路径不存在')
    }
  }

  /**
   * 判断是否是文件
   * @param path
   */
  async isFile(path: string) : Promise<boolean> {
    path = await this.getGeneralPath(path)
    return !await this.isFolder(path)
  }

  /**
   * 获取文件夹下所有文件
   * @param path
   */
  async getChildPaths (path: string) : Promise<Array<string>> {
    path = await this.getGeneralPath(path)
    const files = await fs.readdir(path)
    return files
  }

  /**
   * 读取文件
   * @param path
   * @param encoding
   * @param flag
   */
  async readFile (path: string, encoding?: 'ascii' | 'utf8' | 'utf-8' | 'utf16le' | 'ucs2' | 'ucs-2' | 'base64' | 'base64url' | 'latin1' | 'binary' | 'hex', flag?: string) {
    path = await this.getGeneralPath(path)
    const content = await fs.readFile(path, {
      encoding: encoding,
      flag: flag
    })
    return content
  }

  /**
   * 写入文件
   * @param path
   * @param content
   * @param encoding
   */
  async writFile (path: string, content: string, encoding?: 'ascii' | 'utf8' | 'utf-8' | 'utf16le' | 'ucs2' | 'ucs-2' | 'base64' | 'base64url' | 'latin1' | 'binary' | 'hex') {
    path = await this.getGeneralPath(path)
    await fsExtraPromise.ensureFileSync(path)
    await fs.writeFile(path, content, {
      encoding: encoding
    })
  }

  /**
   * 文件不存在则创建文件
   * @param path
   */
  async ensureFile (path: string) {
    path = await this.getGeneralPath(path)
    await fsExtraPromise.ensureFileSync(path)
  }

  /**
   * 文件夹不存在则创建文件夹
   * @param path
   */
  async ensureFolder (path: string) {
    path = await this.getGeneralPath(path)
    await fsExtraPromise.ensureDir(path)
  }

  /**
   * 设置通用文件路径
   * @param path
   */
  async getGeneralPath (path: string) {
    if (path == null) {
      throw new Error('获取文件路径失败, 文件路径为空')
    }
    return path.replace(/\//g, await this.getSep()).replace(/\\/g, await this.getSep())
  }

  /**
   * 删除文件
   * @param path
   */
  async removeFile (path: string) {
    path = await this.getGeneralPath(path)
    await fsExtraPromise.remove(path)
  }

  /**
   * 获取文件夹下所有文件路径
   * @param path
   */
  async getAllFile (path: string, expands?: Array<string>) : Promise<Array<string>> {
    path = await this.getGeneralPath(path)
    const arr:Array<string> = []
    if (await this.isFolder(path)) {
      let fileArr = await this.getChildFile(path)
      const folderArr = await this.getChildFolder(path)
      if (expands != null) {
        fileArr = await this.filterExpand(fileArr, expands)
      }
      arr.push(...fileArr)
      for (const childPath of folderArr) {
        const childFileArr = await this.getAllFile(childPath, expands)
        arr.push(...childFileArr)
      }
    } else {
      if (expands != null) {
        const filterFileArr = await this.filterExpand([path], expands)
        arr.push(...filterFileArr)
      } else {
        arr.push(path)
      }
    }
    return arr
  }

  /**
   * 获取子文件路径
   * @param path
   */
  async getChildFile (path: string) : Promise<Array<string>> {
    path = await this.getGeneralPath(path)
    const childPaths = await this.getChildPaths(path)
    const results: Array<string> = []
    for (const childPath of childPaths) {
      const absolutePath = path + await this.getSep() + childPath
      if (await this.isFile(absolutePath)) {
        results.push(absolutePath)
      }
    }
    return results
  }

  /**
   * 获取子文件夹路径
   * @param path
   */
  async getChildFolder(path: string) : Promise<Array<string>> {
    path = await this.getGeneralPath(path)
    const childPaths = await this.getChildPaths(path)
    const results: Array<string> = []
    for (const childPath of childPaths) {
      const absolutePath = path + await this.getSep() + childPath
      if (await this.isFolder(absolutePath)) {
        results.push(absolutePath)
      }
    }
    return results
  }

  /**
   * 过滤不需要的文件类型
   * @param paths
   * @param expands
   */
  async filterExpand (paths: Array<string>, expands: Array<string>): Promise<Array<string>> {
    const filterPaths = []
    for (const path of paths) {
      const expand = await this.getFileExpandName(path)
      const index = expands.findIndex(allowExpand => {
        return expand.toLowerCase() === allowExpand.toLowerCase()
      })
      if (index >= 0) {
        filterPaths.push(path)
      }
    }
    return filterPaths
  }

  /**
   * 获取文件写入流
   * @param path
   */
  async getFileWriteStream(path: string) {
    path = await this.getGeneralPath(path)
    await fsExtraPromise.ensureFileSync(path)
    return originFs.createWriteStream(path)
  }

  /**
   * 获取上一级目录
   * @param path
   */
  async getParent (path: string) {
    path = await this.getGeneralPath(path)
    const index = path.lastIndexOf(await this.getSep())
    return path.substring(0, index)
  }

  /**
   * 获取文件前缀
   */
  async getFilePrefix () {
    return 'file:' + await this.getSep() + await this.getSep()
  }

  /**
   * 隐藏文件
   * @param path
   */
  async hideFile (path: string) {
    try {
      path = await this.getGeneralPath(path)
      exec('attrib +s +h ' + path)
    } catch (err) {
      console.error('隐藏文件夹失败')
    }
  }
}
