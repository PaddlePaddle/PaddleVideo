<template>
  <div
    :class="
      'labelitem flex flex-col' + ' ' + customClass + ' ' + activeName + '-box'
    "

    v-if="isUpdate"
  >
    <el-divider content-position="left"
    >标签
      <span
        class="items-center inline-flex justify-between h-10 py-1 px-3 bg-white rounded-lg text-gray-600"
      >
        <div class="hover:shadow-md h-4 box">
          <span
            class="iconfont icon-item icon-biaoqian1 cursor-pointer"
            @click="changeActive('label')"
            :class="{ active: activeName == 'label' }"
          /></div><el-divider direction="vertical"/>
                <div class="hover:shadow-md h-4">
        <span
          class="iconfont icon-item icon-bianji1 cursor-pointer"
          @click="changeActive('modify')"
          :class="{ active: activeName == 'modify' }"
        /></div>
      </span>
    </el-divider>
    <div class="flex flex-col space-y-5">
      <el-input v-model="filterText" placeholder="查找标签"/>
      <el-tree
        ref="treeRef"
        :data="labeldefine"
        draggable
        node-key="id"
        show-checkbox
        :filter-node-method="filterNode"
        @check-change="handleCheckChange"
        :expand-on-click-node="false"
        default-expand-all
      >
        <template #default="{ node, data }">
          <!--          {{ data.indeterminate || data.check }}-->
          <span
            class="relative label-wrap-box flex items-center -left-6 space-x-0.5 space-y-2.5"
          >
            <label
              v-if="label"
              v-show="activeName == 'label'"
              class="w-4 h-4 rounded-sm border-2 border-solid relative cursor-pointer place-content-center flex"
              :style="{
                borderColor: data.color,
                backgroundColor:
                  data.indeterminate || data.check
                    ? data.color
                    : 'rgba(255, 69, 0, 0)'
              }"
            >
              <span
                :class="
                  data.indeterminate
                    ? 'indeterminate'
                    : data.check
                    ? 'hook'
                    : ''
                "
              />
            </label>
            <div
              v-show="activeName == 'modify'"
              :style="{ backgroundColor: data.color }"
              class="w-1 h-4"
            ></div>
            <span
              @dblclick="handleDblClick(data)"
              @keyup.enter.native="data.edit = false"
            >
              <span v-if="!data.edit">{{ data.label }}</span>
              <div class="w-20">
                <el-input
                  ref="edit"
                  v-if="data.edit"
                  size="small"
                  v-model="data.label"
                  @blur="data.edit = false"
                />
              </div>
            </span>
            <span
              v-if="modify"
              class="flex space-x-2"
              v-show="activeName == 'modify'"
            >
              <p
                class="flex-1 h-full text-xs text-center text-gray-600"
              >
                {{ getStatisticsLabels(data.id) }}
              </p>
              <!--          <div class="label-item flex flex-col float-right">-->
             <el-dropdown :hide-on-click="false">
              <span class=" iconfont icon-more  cursor-pointer"/>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item>
                    <div>
                      <el-color-picker
                        v-model="data.color"
                        show-alpha
                        :predefine="predefineColors"
                      />
                      <span class="iconfont icon-tiaoshi"></span> 修改颜色
                    </div>
                  </el-dropdown-item>
                  <el-dropdown-item>
                    <div @click="add(node)">
                      <span class="iconfont icon-information_add"></span>
                      添加同级标签
                    </div>
                  </el-dropdown-item>
                  <el-dropdown-item><div @click="append(data)">
                  <span class="iconfont icon-fangkuai"></span>
                                        添加子级标签
                </div></el-dropdown-item>
                  <el-dropdown-item><div @click="remove(node, data)">
                  <span class="iconfont icon-fangkuai-"></span>
                  删除标签
                </div></el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
              <!--          </div>-->
          </span>
          </span>
        </template>
      </el-tree>
      <div class="flex place-content-evenly" v-show="activeName == 'label'">
        <div
          class="submit hover:shadow-indigo-500/50 hover:text-white hover:shadow-md hover:bg-indigo-600 bg-indigo-600 bg-opacity-90 shadow button inline-flex relative items-center cursor-pointer justify-center h-9 px-5 py-2 rounded-lg"
          @click="onSubmit"
        >
          <p class="text-xs font-semibold text-center text-white">确定</p>
        </div>
        <div
          class="hover:shadow-md hover:bg-red-50 button inline-flex relative items-center cursor-pointer justify-center h-9 px-5 py-2 bg-white shadow rounded-lg"
          @click="resetChecked"
        >
          <p class="text-xs font-semibold text-center  text-red-500">清空</p>
        </div>
      </div>
    </div>
    <el-divider content-position="left" v-show="activeName == 'label'"
    >时段
    </el-divider
    >
    <div
      class="inline-flex place-content-evenly items-center"
      v-show="activeName == 'label'"
    >
      <p class="w-6 h-4 text-xs text-center text-gray-500">开始</p>
      <div class="w-1/3">
        <el-input
          v-model="curLabel.range[0]"
          placeholder=""
          :suffix-icon="Timer"
        />
      </div>
      <p class="w-6 h-4 text-xs text-center text-gray-500">结束</p>
      <div class="w-1/3">
        <el-input
          v-model="curLabel.range[1]"
          placeholder=""
          :suffix-icon="Timer"
        />
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import {Card} from '/@/components'
import {useService} from '/@/composables'
import {computed, nextTick, onMounted, onUnmounted, reactive, ref, toRefs, watch} from 'vue'
import {LABELDEFINE, useLabelsStore, useLabelDefineStore, useProject} from '/@/store'
import {Timer} from '@element-plus/icons-vue'
import {storeToRefs} from 'pinia'
import {useDebounceFn} from '@vueuse/core'
import {bus} from '/@/utils/busUtils'
import {CHANGE_LABEL_ACTIVE, CHANGE_LABEL_DEFINE} from '/@/constants'
import {currentExportJson} from '/@/utils/projectUtil'

const {getSep} = useService('FileService')
const Label = useLabelsStore()
const LabelDefine = useLabelDefineStore()
const {submitLabel} = Label
let labeldefine = LabelDefine.labeldefine
const {curLabel} = storeToRefs(Label)
const Project = useProject()
const isUpdate = ref(true)
const props = defineProps({
  label: {type: Boolean, default: true},
  normal: {type: Boolean, default: true},
  modify: {type: Boolean, default: true},
  customClass: String
})
// 比较奇怪，数据已经变了，但是视图没有刷新
bus.on(CHANGE_LABEL_DEFINE, function () {
  labeldefine = LabelDefine.labeldefine
})

const activeName = ref('label')

function changeActive(val: string) {
  activeName.value = val
  bus.emit(CHANGE_LABEL_ACTIVE, val)
}

bus.on(CHANGE_LABEL_ACTIVE, (val: string) => {
  activeName.value = val
})

const edit = ref()
const handleDblClick = (data) => {
  if (activeName.value == 'normal' || activeName.value == 'label') {
    return
  }
  data.edit = true
  nextTick(() => {
    edit.value.input.focus()
  })
}
const filterText = ref('')
const treeRef = ref<InstanceType<typeof ElTree>>()
watch(filterText, (val) => {
  treeRef.value!.filter(val)
})

function treeToArray(tree: Array<LABELDEFINE>): Array<LABELDEFINE> {
  let res: Array<LABELDEFINE> = []
  for (const item of tree) {
    const {children, ...i} = item
    res.push(i)
    if (children && children.length) {
      res = res.concat(treeToArray(children))
    }
  }
  return res
}

class Child {
  constructor() {
    this.id = new Date().getTime().toString()
    this.color = '#' + Math.random().toString(16).substr(2, 6).toUpperCase()
    this.label = '新建标签'
    this.edit = true
    this.total = 0
    this.indeterminate = false
    this.check = false
    this.children = []
  }
}

const append = (data: LABELDEFINE | null) => {
  const newChild = new Child()
  if (!data) treeRef.value.append(newChild)
  else {
    data.children.push(JSON.parse(JSON.stringify(newChild)))
  }
  nextTick(() => {
    edit.value.input.focus()
  })
}
const add = (data: Node) => {
  const newChild = new Child()
  const parent = data.parent.data
  if (!parent) treeRef.value.append(newChild)
  else {
    parent.children.push(JSON.parse(JSON.stringify(newChild)))
  }
  nextTick(() => {
    edit.value.input.focus()
  })
}

const remove = (node: Node, data: LABELDEFINE) => {
  const parent = node.parent
  const children: LABELDEFINE[] = parent.data.children || parent.data
  const index = children.findIndex((d) => d.id === data.id)
  children.splice(index, 1)
  if (!treeRef.value.data.length) {
    append()
  }
}

const filterNode = (value: string, data: LABELDEFINE) => {
  if (!value) return true
  return data.label.indexOf(value) !== -1
}
const handleCheckChange = (
  node: Object,
  checked: boolean,
  indeterminate: boolean
) => {
  node.indeterminate = indeterminate
  node.check = checked
  if (Project.project.baseConfig.autoSave && !submited.value) submitLabel()
}
const setCheckedKeys = (key) => {
  treeRef.value!.setCheckedKeys(key, false)
}
const setCheckedNodes = (node) => {
  treeRef.value!.setCheckedNodes(node, false)
}
const getCheckedNodes = () => {
  if (treeRef.value == null) {
    return []
  }
  return treeRef.value!.getCheckedNodes(false, false)
}

async function resetChecked() {
  const keys = []
  for await (const data of treeRef.value.data) {
    keys.push(data.id.toString())
  }
  await setCheckedKeys(keys)
  treeRef.value!.setCheckedKeys([], false)
}

nextTick(() => resetChecked())
const onSubmit = async () => {
  submitLabel()
  const project = useProject()
  if (project.baseConfig.autoSave) {
    const jsonDefaultPath =
      project.baseConfig.projectPath +
      (await getSep()) +
      project.baseConfig.projectName +
      '.pConfig'
    await currentExportJson(jsonDefaultPath)
  }
}
nextTick(() => resetChecked())
watch(
  () => curLabel.value,
  (newc, oldc) => {
    // console.log('oldc=======================', oldc)
    if (oldc) oldc.edit = false
    if (newc && newc.label) {
      nextTick(() => setCheckedNodes(newc.label))
    }
  }
)
nextTick(() =>
  watch(
    () => getCheckedNodes(),
    (v) => {
      if (curLabel.value) {
        curLabel.value.label = v
          if (v.length) {
            curLabel.value.color = v[0].color
          } else curLabel.value.color = 'rgba(255,255,255,1)'
        if (!curLabel.value.saved) {
          submitLabel()
        }
      }
    }
  )
)

const predefineColors = reactive([
  '#ff4500',
  '#ff8c00',
  '#ffd700',
  '#90ee90',
  '#00ced1',
  '#1e90ff',
  '#c71585',
  'rgba(255, 69, 0, 0.68)',
  'rgb(255, 120, 0)',
  'hsv(51, 100, 98)',
  'hsva(120, 40, 94, 0.5)',
  'hsl(181, 100%, 37%)',
  'hsla(209, 100%, 56%, 0.73)',
  '#c7158577'
])

const keyCodeArr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
const keydownEvent = useDebounceFn((event: any) => {
  const index = keyCodeArr.findIndex((item) => {
    return item.toString() === event.key
  })
  if (index >= 0) {
    const treeArr = treeToArray(labeldefine)
    const keyCode = keyCodeArr[index]
    if (keyCode > treeArr.length) {
      return
    }
    const selectItem = treeArr[keyCode - 1]
    treeRef.value!.setChecked(selectItem, !getIsCheck(selectItem), true)
  }
}, 200)

function getIsCheck(node: any) {
  const checkArr = treeRef.value!.getCheckedNodes()
  const index = checkArr.findIndex((item: any) => {
    return item.uuid === node.uuid
  })
  return index >= 0
}

function getStatisticsLabels(id) {
  const labels = Label.getLabels()
  const curVideo = Project.curVideo
  let count = 0
  // console.log(labels)

  if (curVideo && labels) {
    const index = labels.findIndex(item => {
      return item.uuid == curVideo
    })
    if (index >= 0) {
      const nowLabel = labels[index]
      for (const videoxLabel of nowLabel.videoxLabels) {
        for (const labelElement of videoxLabel.label) {
          if (labelElement.id === id) {
            count++
          }
        }
      }
    }
  }
  return count
}

onMounted(() => {
  window.removeEventListener('keypress', keydownEvent)
  window.addEventListener('keypress', keydownEvent)
})
onUnmounted(() => {
  window.removeEventListener('keypress', keydownEvent)
})
</script>

<style>
.labelitem {
  font-size: 16px;
}

.normal-box.labelitem i + label {
  display: none;
}

.modify-box.labelitem i + label {
  display: none;
}

.normal-box .label-wrap-box {
  left: 0;
}

.modify-box .label-wrap-box {
  left: 0;
}

.labelitem i + label {
  /*@apply z-10  ;*/
  @apply z-10 opacity-0;
}

.labelitem .icon-item:hover {
  @apply text-indigo-600 !important;
}

.labelitem .icon-item.active {
  @apply text-indigo-600  !important;
}

.labelitem .icon-item.active {
  @apply shadow-md  !important;
}

.el-color-picker__trigger {
  @apply w-full !important;
  /*@apply z-10 opacity-0 left-4 w-4 h-4 overflow-hidden !important;*/
}

.el-color-picker {
  @apply z-10 w-28 h-4 overflow-hidden absolute opacity-0 !important;
  /*@apply z-10 opacity-0 left-4 w-4 h-4 overflow-hidden !important;*/
}

.el-tree {
  @apply bg-transparent !important;
}

.labelitem .el-tree-node__content {
  @apply hover:bg-white hover:bg-opacity-20 !important;
}

/*hover:bg-amber-300*/
.hook {
  opacity: 1;
  position: absolute;
  width: 5px;
  height: 9px;
  background: transparent;
  top: 0px;
  right: 3px;
  border: 1px solid #fff;
  border-top: none;
  border-left: none;
  -webkit-transform: rotate(35deg);
  -moz-transform: rotate(35deg);
  -o-transform: rotate(35deg);
  -ms-transform: rotate(35deg);
  transform: rotate(35deg);
}

.indeterminate {
  opacity: 1;
  width: 7px;
  height: 1px;
  background: transparent;
  border: 1px solid #fff;
  border-top: none;
  border-left: none;
}

.label-item .label-tip {
  border-radius: 5px !important;
  transform: translateX(-50%);
  left: 50%;
  margin-top: 0 !important;
}

.label-item {
  position: relative;
  white-space: nowrap;
}

.label-item:hover .label-tip {
  @apply visible;
}

.label-item .label-tip {
  @apply hover:visible;
  /*@apply invisible hover:visible;*/
}
</style>

