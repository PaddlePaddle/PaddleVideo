import { defineStore, mapActions, mapState, mapStores } from 'pinia'
import { useStorage } from '@vueuse/core'
import { computed, nextTick, ref, watch } from 'vue'
import { bus } from '/@/utils/busUtils'
import { CHANGE_LABEL_DEFINE } from '/@/constants'

export interface LABELDEFINE {
  id: number
  label: string
  color: string
  total: number
  check: boolean
  edit: boolean
  indeterminate: boolean
  children?: LABELDEFINE[]
}

const _labeldefine: LABELDEFINE[] = [
  {
    id: 1,
    total: 0,
    color: '#ff4500',
    label: '进球',
    edit: false,
    indeterminate: false,
    check: false,
    children: [
      {
        id: 4,
        total: 0,
        color: 'rgba(72, 169, 167, 0.68)',
        label: '三分球',
        edit: false,
        indeterminate: false,
        check: false,
        children: []
      }, {
        id: 9,
        total: 0,
        color: '#c71585',
        edit: false,
        indeterminate: false,
        check: false,
        label: '二分球'
      }
    ]
  }
]

export const useLabelDefineStore = defineStore('LabelDefine', () => {
  const labeldefine = useStorage('labeldefine', _labeldefine)
  function setLabelDefine (data) {
    console.log('set label define')
    labeldefine.value = data
  }
  watch(labeldefine, function () {
    bus.emit(CHANGE_LABEL_DEFINE)
  })
  return {
    labeldefine,
    setLabelDefine
  }
})
