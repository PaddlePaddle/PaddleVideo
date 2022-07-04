import { defineStore, mapActions, mapState, mapStores } from 'pinia'
import { useStorage } from '@vueuse/core'
import { computed, nextTick, reactive, ref, toRaw, toRefs, unref, UnwrapNestedRefs, watch } from 'vue'
import { useProject } from '/@/store/project'
import { cloneDeep } from 'lodash'
import { bus } from '/@/utils/busUtils'
import { CHANGE_DOUBLE_CLICK_INDEX, CHANGE_PLAY_CURRENT_TIME } from '/@/constants'

export const useLabelsStore = defineStore('Labels', () => {
  const Project = useProject()
  const curVideo = computed(() => Project.project.curVideo)
  const projectLabels = useStorage('projectLabels', {})
  const curLabel = ref({ range: [0, 0] })
  const selectedLabel = ref()

  function initLabel() {
    const newLabel = reactive({
      uuid: (new Date().getTime()).toString(),
      range: [],
      label: [],
      color: ref(),
      edit: false, // 双击修改切换状态
      saved: false
    })
    if (!projectLabels.value[curVideo.value]) {
      projectLabels.value[curVideo.value] = []
    }
    projectLabels.value[curVideo.value].push(newLabel)
    return newLabel
  }

  function submitLabel() {
    if (curLabel.value.saved) return
    curLabel.value.saved = true
  }
  function getTime(v) {
    const duration = (Project.videoMeta(Project.project.curVideo as string) as any).duration
    return Math.floor((duration * (v / 100)))
  }
  function setdotsPos(v) {
    if (!curLabel.value) curLabel.value = initLabel()
    if (curLabel.value.range.length === 2) {
      submitLabel()
      curLabel.value = initLabel()
    }
    const scale = getTime(v)
    curLabel.value.range.push(scale)
    bus.emit(
      CHANGE_PLAY_CURRENT_TIME,
      scale
    )
    if (curLabel.value.range.length === 2) {
      if (Project.project.baseConfig.autoSave) {
        submitLabel()
        curLabel.value = initLabel()
      }
    }
  }
  let DOUBLE_CLICK: boolean = true
  bus.on(CHANGE_DOUBLE_CLICK_INDEX, (flag) => {
    // console.log('CHANGE_DOUBLE_CLICK_INDEX', flag)
    DOUBLE_CLICK = flag as boolean
  })
  const changeSegment = (e, pos, uuid) => {
    if (DOUBLE_CLICK && curLabel.value && !curLabel.value.edit) return
    const duration = (Project.videoMeta(Project.project.curVideo as string) as any).duration
    const scale = Math.floor((duration * (e / 100)) * 100) / 100
    getSegment(uuid)
    curLabel.value.range[pos] = scale
  }

  const getSegment = (uuid) => {
    curLabel.value = projectLabels.value[curVideo.value].find((item) => item.uuid === uuid)
  }

  function getcurLabelIdex() {
    return projectLabels.value[curVideo.value].findIndex((item) => item.uuid == curLabel.value.uuid)
  }

  function getLabelIdexById(uuid) {
    return projectLabels.value[curVideo.value].findIndex((item) => item.uuid == uuid)
  }

  function deletecurLabel() {
    if (selectedLabel.value) {
      selectedLabel.value.forEach(item => {
        const index = getLabelIdexById(item.uuid)
        deleteLabel(index)
      })
    } else if (curLabel.value) {
      const index = getcurLabelIdex()
      deleteLabel(index)
    }
  }

  function deleteLabelById(uuid) {
    const index = getLabelIdexById(uuid)
    deleteLabel(index)
  }

  function deleteLabel(index) {
    if (index !== -1) {
      projectLabels.value[curVideo.value].splice(index, 1)
      if (projectLabels.value[curVideo.value].length) {
        if (projectLabels.value[curVideo.value].length === index) {
          curLabel.value = projectLabels.value[curVideo.value][index - 1]
        } else {
          curLabel.value = projectLabels.value[curVideo.value][index]
        }
      }
    }
  }

  function getLabels() {
    if (curVideo.value) {
      return projectLabels.value[curVideo.value]
    } else return []
  }

  function getAllLabels () {
    const videoList = cloneDeep(Project.project.videoList)
    for (const videoListElement of videoList) {
      if (projectLabels.value[videoListElement.uuid]) {
        videoListElement.labelList = projectLabels.value[videoListElement.uuid]
      } else {
        videoListElement.labelList = []
      }
    }
    return videoList
  }

  function resetProjectLabels() {
    projectLabels.value = {}
  }

  function setProjectLabels(data) {
    projectLabels.value = data
  }

  return {
    curLabel,
    selectedLabel,
    curVideo,
    getLabels,
    getAllLabels,
    deletecurLabel,
    deleteLabelById,
    setdotsPos,
    getSegment,
    submitLabel,
    changeSegment,
    resetProjectLabels,
    setProjectLabels,
    getTime
  }
})
