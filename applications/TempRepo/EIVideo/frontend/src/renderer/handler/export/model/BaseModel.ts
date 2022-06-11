import { exportParams } from '../index'
import { useProject } from '/@/store'
export default class BaseModel {
  async genModel(params: exportParams) {
    if (params.filterArr == null) {
      const project = useProject()
      params.filterArr = project.exportConfigArr
    }
    let result = pregetRealRange(params.data)
    result = await this.preFilterModel(result, params.filterArr)
    result = await this.customGenModel(result)
    return result
  }

  async customGenModel (input: any) {
    return input
  }

  async preFilterModel (input: any, filterArr: any) {
    const arr = []
    const project = useProject()
    for (const item of input) {
      const resultMap = getItem(null, item, filterArr)
      if (project.baseConfig.outContent === 'all') {
        arr.push(resultMap)
      }
      if (project.baseConfig.outContent === 'viewed') {
        const index = project.videoList.findIndex(item => {
          return item.uuid == resultMap.uuid
        })
        if (index >= 0) {
          if (project.videoList[index].isPlayed) {
            arr.push(resultMap)
          }
        }
      }
      if (project.baseConfig.outContent === 'labeled') {
        if (resultMap.videoxLabels.length > 0) {
          arr.push(resultMap)
        }
      }
    }
    return arr
  }
}

function getItem(groupName: string | null, input: any, filterArr: any) {
  const resultMap = {}
  for (const inputKey in input) {
    const configItem = findConfigByKey(groupName, inputKey, filterArr)
    if (configItem == null) {
      continue
    }
    if (!getIsNeedAdd(configItem)) {
      continue
    }
    if (configItem.multiply || (configItem.children && configItem.children.length > 0)) {
      const childrenItem = input[inputKey]
      if (Array.isArray(childrenItem)) {
        const childrenArr = []
        for (const childrenItemElement of childrenItem) {
          const childrenMap = getItem(configItem.validateId || configItem.id, childrenItemElement, filterArr)
          childrenArr.push(childrenMap)
        }
        // @ts-ignore
        resultMap[configItem.newId] = childrenArr
      } else {
        const childrenMap = getItem(configItem.validateId || configItem.id, childrenItem, filterArr)
        // @ts-ignore
        resultMap[configItem.newId] = childrenMap
      }
    } else {
      // @ts-ignore
      resultMap[configItem.newId] = input[inputKey]
    }
  }
  return resultMap
}

function findConfigByKey (groupName: string | null, key: string, filterArr: any) {
  if (groupName == null) {
    groupName = 'video'
  }
  const exportConfigArr = treeToArray(filterArr)
  const index = exportConfigArr.findIndex(item => {
    return item.group == groupName && item.id == key
  })
  if (index >= 0) {
    return exportConfigArr[index]
  }
  return null
}

function treeToArray(tree: Array<any>): Array<any> {
  let res: Array<any> = []
  for (const item of tree) {
    const { children } = item
    res.push(item)
    if (children && children.length) {
      res = res.concat(treeToArray(children))
    }
  }
  return res
}

function getIsNeedAdd (item: any) {
  if (item.check) {
    return true
  }
  return findHasChildrenAdd(item)
}

function findHasChildrenAdd (item: any) {
  if (item.children == null || item.children.length === 0) {
    return false
  }
  for (const child of item.children) {
    if (child.check === true) {
      return true
    } else {
      if (findHasChildrenAdd(child)) {
        return true
      }
    }
  }
  return false
}

function pregetRealRange (input) {
  for (const item of input) {
    const videoUuid = item.uuid
    const labels = item.labelList
    for (const label of labels) {
      if (label.range.size > 1) {
        label.range = [videoUuid, label.range[0], label.range[1]]
      }
    }
    item.videoxLabels = labels
  }
  input = input.filter(item => {
    return item.uuid != null
  })
  return input
}

function getRealRange(videoUuid, number) {
  if (number == null || number === '') {
    return 0
  }
  const project = useProject()
  const duration = project.videoMeta(videoUuid).duration
  const scale = Math.floor((duration * (number / 100)))
  return scale
}
