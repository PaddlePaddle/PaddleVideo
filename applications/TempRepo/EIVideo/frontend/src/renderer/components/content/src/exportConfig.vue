<template>
  <div class="export-config-box">
    <el-tree
      ref="tree"
      :allow-drop="allowDrop"
      :data="treeData"
      draggable
      show-checkbox
      node-key="uuid"
      default-expand-all
      @check-change="handleCheckChange"
    >
      <template #default="{ node, data }">
        <span class="custom-tree-node">
          <label
            class="w-4 h-4 rounded-sm border-2 border-solid relative cursor-pointer place-content-center flex mr-1"
            :style="{
              minWidth: '1rem',
              borderColor: '#4b44ee',
              // borderColor: data.color,
              backgroundColor:
                data.indeterminate || data.check
                  ? '#4b44ee'
                  // ? data.color
                  : 'rgba(255, 69, 0, 0)'
            }"
          >
            <span
              :class="
                data.indeterminate ? 'indeterminate' : data.check ? 'hook' : ''
              "
            />
          </label>
          <span v-if="!data.isAddEditor"
            >{{ data.name }}({{ data.id }}) -
          </span>
          <el-select
            ref="select"
            style="width: 100px"
            v-model="data.optionValue"
            v-if="data.isAddEditor"
            @blur="addFinish(data)"
          >
            <el-option
              :id="item.name"
              :key="item.id"
              :value="item"
              :label="item.name"
              v-for="item in getGroupOption(data.group)"
            ></el-option>
          </el-select>
          <span v-if="!data.isChangeNewId" @click="handleChangeId(data)">{{
            data.newId
          }}</span>
          <el-input
            v-else
            v-model="data.newId"
            ref="edit"
            :autofocus="true"
            @blur="data.isChangeNewId = !data.isChangeNewId"
          ></el-input>
        </span>

        <span class="flex place-content-center space-x-0.5">
          <a
            @click="add(node)"
            v-show="getIsCanAdd('video') && data.group == 'video'"
          >
            <span class="iconfont icon-information_add"></span> </a
          ><a
            @click="append(data)"
            v-show="getIsCanAdd(data.group) && data.group != 'video'"
          >
            <span class="iconfont icon-fangkuai"></span>
          </a>
          <a @click="remove(data)" v-show="data.isNewAdd">
            <span class="iconfont icon-fangkuai-"></span>
          </a>
        </span>
      </template>
    </el-tree>
    <el-card class="view-box text-left mt-5">
      <json-viewer copyable expanded :expand-depth='99' :value="jsonData"></json-viewer>
    </el-card>
  </div>
</template>

<script lang="ts" setup>
import { nextTick, onMounted, Ref, ref, defineExpose, watch } from 'vue'
import JsonViewer from 'vue-json-viewer'
import type { DropType } from 'element-plus/es/components/tree/src/tree.type'
import { v1 as uuidv4 } from 'uuid'
import { useLabelsStore, useProject } from '/@/store'
import { ElNotification } from 'element-plus'
import { cloneDeep } from 'lodash'
import { getExportModel } from '/@/handler/export'
import { useDebounceFn } from '@vueuse/core'
const labelsStore = useLabelsStore()
// 注意这里配置label是一个递归结构，所以有findId 修改查找ID，validateId修改校验ID，multiply是否有子元素
const exportConfig = {
  video: [
    {
      id: 'videoName',
      name: '视频名称',
      isNeedSelect: false
    },
    {
      id: 'uuid',
      name: '视频编号',
      isNeedSelect: false
    },
    {
      id: 'duration',
      name: '时长'
    },
    {
      id: 'size',
      name: '大小'
    },
    {
      id: 'videoxLabels',
      name: '标签',
      isNeedSelect: false
    }
  ],
  videoxLabels: [
    {
      id: 'uuid',
      name: '标签编号',
      isNeedSelect: false
    },
    {
      id: 'label',
      name: '标签列表',
      isNeedSelect: false
    },
    {
      id: 'range',
      name: '范围',
      isNeedSelect: false
    },
    {
      id: 'color',
      name: '颜色'
    }
  ],
  label: [
    {
      id: 'id',
      name: '编号',
      isNeedSelect: false
    },
    {
      id: 'label',
      findId: 'labelName',
      name: '标签名',
      isNeedSelect: false
    },
    {
      id: 'color',
      name: '颜色'
    },
    {
      id: 'children',
      multiply: true,
      validateId: 'label',
      name: '子标签'
    }
  ]
}
const tree: Ref = ref()
const treeData: Ref = ref([])
const edit = ref()
const select = ref()
const jsonData: Ref = ref()

const handleChangeId = (data: any) => {
  data.isChangeNewId = !data.isChangeNewId
  console.log(edit.value)
  nextTick(() => {
    edit.value.input.focus()
  })
}
watch(
  treeData,
  function () {
    console.log('change')
    renderJsonModel()
  },
  {
    deep: true
  }
)

const renderJsonModel = useDebounceFn(async () => {
  const Label = useLabelsStore()
  const labels = Label.getAllLabels()
  const treeArray = labels
  const jsonModel = await getExportModel({
    modelType: 'FF',
    outputType: 'json',
    data: treeArray,
    filterArr: treeData.value,
    encoding: null
  })
  const data = JSON.parse(jsonModel as string)
  nextTick(() => {
    jsonData.value = data
  })
}, 1000)

const getGroupOption = (name: string) => {
  const treeArr = treeToArray(treeData.value)
  // @ts-ignore
  const groupArr = exportConfig[name]
  const arr = []
  for (const groupArrElement of groupArr) {
    const index = treeArr.findIndex((item) => {
      return item.name === groupArrElement.name
    })
    if (index < 0) {
      arr.push(groupArrElement)
    }
  }
  return arr.map((item) => {
    return {
      id: item.id,
      name: item.name,
      label: item.name,
      findId: item.findId,
      multiply: item.multiply,
      validateId: item.validateId
    }
  })
}

const addFinish = (data: any) => {
  console.log(data.optionValue)
  setTimeout(() => {
    if (data.optionValue == null) {
      tree.value!.remove(data.uuid)
    } else {
      console.log(data.optionValue)
      data.id = data.optionValue.id
      data.findId = data.optionValue.findId
      data.multiply = data.optionValue.multiply
      data.name = data.optionValue.name
      data.newId = data.optionValue.id
      data.validateId = data.optionValue.validateId
      data.isAddEditor = false
    }
  }, 100)
}

const remove = (data: any) => {
  tree.value!.remove(data.uuid)
}

const handleCheckChange = (
  node: any,
  checked: boolean,
  indeterminate: boolean
) => {
  node.indeterminate = indeterminate
  node.check = checked
}

const project = useProject()
onMounted(() => {
  let arr = []
  if (project.exportConfigArr.length === 0) {
    arr = initTree('video')
    project.changeExportConfigArr(arr)
  } else {
    arr = cloneDeep(project.exportConfigArr)
  }
  treeData.value.push(...arr)
  initCheck()
})

function resetConfig() {
  treeData.value = initTree('video')
}

function confirm() {
  const treeArr = treeToArray(treeData.value)
  const index = treeArr.findIndex((item) => {
    return item.check
  })
  if (index >= 0) {
    console.log(treeData.value)
    project.changeExportConfigArr(treeData.value)
    return true
  } else {
    ElNotification({
      message: '请至少选择一个导出项',
      type: 'warning'
    })
    return false
  }
}

function initCheck() {
  const treeArr = treeToArray(treeData.value)
  const checkArr = treeArr
    .filter((item) => {
      return item.check
    })
    .map((item) => {
      return item.uuid
    })
  console.log(checkArr)
  nextTick(() => {
    tree.value!.setCheckedKeys(checkArr, false)
  })
}

function show() {
  treeData.value = []
  console.log(project.exportConfigArr)
  const arr = cloneDeep(project.exportConfigArr)
  treeData.value.push(...arr)
  initCheck()
  renderJsonModel()
}

defineExpose({
  resetConfig,
  confirm,
  show
})

const add = (node: any) => {
  const flag = getIsCanAdd('video')
  if (!flag) {
    return null
  }
  const item = getNewTreeItem('video')
  treeData.value.push(item)
  nextTick(() => {
    select.value.focus()
  })
}

const append = (data: any) => {
  const flag = getIsCanAdd(data.group)
  if (!flag) {
    return null
  }
  const item = getNewTreeItem(data.group)
  const treeArr = treeToArray(treeData.value)
  const index = treeArr.findIndex((item: any) => {
    return item.id === data.group
  })
  if (index < 0) {
    return
  }
  treeArr[index].children.push(item)
  nextTick(() => {
    select.value.focus()
  })
}

const allowDrop = (draggingNode: any, dropNode: any, type: DropType) => {
  console.log(dropNode.data)
  if (type === 'inner') {
    return false
  }
  if (draggingNode.data.group === dropNode.data.group) {
    return true
  }
  return false
}

function initTree(name: string) {
  const arr = []
  // @ts-ignore
  for (const videoElement of exportConfig[name]) {
    if (videoElement.isNeedSelect === false) {
      const treeItem = getTreeItem(name, videoElement)
      getTreeItemChildren(treeItem)
      arr.push(treeItem)
    }
  }
  return arr
}

function getTreeItem(name: string, item: any) {
  return {
    uuid: uuidv4(),
    id: item.id,
    validateId: item.validateId,
    multiply: item.multiply,
    findId: item.findId,
    group: name,
    name: item.name,
    newId: item.id,
    isChangeNewId: false,
    color: '#' + Math.random().toString(16).substr(2, 6).toUpperCase(),
    indeterminate: false,
    check: true,
    isNewAdd: false,
    isAddEditor: false,
    optionValue: null,
    children: []
  }
}

function getIsCanAdd(name: string) {
  return getGroupOption(name).length > 0
}

function getNewTreeItem(name: string) {
  return {
    uuid: uuidv4(),
    id: null,
    validateId: null,
    multiply: false,
    findId: null,
    group: name,
    name: null,
    newId: null,
    isChangeNewId: false,
    color: '#' + Math.random().toString(16).substr(2, 6).toUpperCase(),
    indeterminate: false,
    check: false,
    isNewAdd: true,
    isAddEditor: true,
    optionValue: null,
    children: []
  }
}

function getTreeItemChildren(item: any) {
  const findId = item.findId == null ? item.id : item.findId
  // @ts-ignore
  if (exportConfig[findId] == null) {
    return
  }
  const children = initTree(findId)
  item.children.push(...children)
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
</script>

<style>
.export-config-box {
  @apply w-full;
}
.export-config-box i + label {
  @apply opacity-0 z-10;
}
.export-config-box .custom-tree-node {
  @apply flex items-center relative -left-6;
}
.export-config-box .el-tree-node__content {
  @apply hover:bg-white hover:bg-opacity-20 !important;
}
</style>
