import { defineStore, mapActions, mapState, mapStores } from 'pinia'
import { useStorage } from '@vueuse/core'
import { computed, nextTick, ref } from 'vue'
export interface OutputDefine {
  id: number
  key: string
  name: string
  check: boolean
  edit: boolean
  indeterminate: boolean
  children?: OutputDefine[]
}
const _outputdefine: OutputDefine = [
  {
    id: 1,
    key: 'uuid',
    name: '数据编号',
    outName: 'uuid',
    edit: false,
    indeterminate: false,
    check: false,
    children: []
  }, {
    id: 2,
    key: 'range',
    name: '时间段',
    outName: 'range',
    edit: false,
    indeterminate: false,
    check: false,
    children: []
  }, {
    id: 3,
    key: 'label',
    name: '标签',
    outName: 'label',
    edit: false,
    indeterminate: false,
    check: false,
    children: [{
      id: 4,
      key: 'label',
      name: '标签名',
      outName: 'labelName',
      edit: false,
      indeterminate: false,
      check: false,
      children: []
    }, {
      id: 5,
      key: 'id',
      name: '标签编号',
      outName: 'labelId',
      edit: false,
      indeterminate: false,
      check: false,
      children: []
    }]
  }
]

export const useOutputDefineStore = defineStore('OutputDefine', () => {
  const outputdefine = useStorage('outputdefine', _outputdefine)
  return {
    outputdefine
  }
})
