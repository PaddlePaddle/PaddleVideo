<template>
  <div style="min-width: 1px" class="h-screen">
    <transition name="browser-fade">
      <div class='relative' ref='browserBox' v-show="project.isBrowserShow"
           :style='{width: boxWidth + "px"}'>
        <div
          :style='{width: boxWidth + "px"}'
          class="browser-box px-4 py-2 bg-white shadow overflow-y-auto rounded-2xl h-screen"
        >
          <el-tabs v-model="activeName">
            <el-tab-pane name="img">
              <template #label>
                <span class="iconfont icon-icon_yingyongguanli"/>
              </template>
              <ImageBrowser :filterValue="state"></ImageBrowser>
            </el-tab-pane>
            <el-tab-pane name="file">
              <template #label>
                <span class="iconfont icon-liebiao"/>
              </template>
              <FileBrowser :filterValue="state"></FileBrowser>
            </el-tab-pane>
            <el-tab-pane disabled>
              <template #label>
                <p
                  class="text-xs font-medium leading-none text-center text-gray-500"
                >
                  <el-input
                    v-model="state"
                    placeholder="查找文件"
                    @select="handleSelect"
                  />
                </p>
              </template>
            </el-tab-pane>
          </el-tabs>
        </div>
        <div class='border-right-box'  @mousedown='down'></div>
      </div>
    </transition>
  </div>
  <span
    class="hover:text-indigo-600 font-bold iconfont icon-zhankai cursor-pointer pl-2 py-2 panel-control-button"
    @click="setBrower"
  ></span>
</template>
<style>
.border-right-box{
  height: 100%;
  width: 2px;
  background: #eee;
  cursor: e-resize;
  position: absolute;
  right: 0;
  top: 0;
}
.browser-box {
  width: 300px;
  z-index: 999;
}

.browser-box .el-tabs {
  width: 100%;
}

.browser-box .el-tab-pane {
  min-width: 100%;
  display: flex;
}

.light-off .browser-box .el-tab-pane {
  z-index: 999;
}

.browser-box .el-tabs__content {
  display: flex;
  overflow: visible !important;
}

.light-off .browser-box .el-tabs__content {
  z-index: 999;
}

.el-tabs__nav-wrap::after {
  @apply w-0;
}

.el-tabs__active-bar {
  @apply w-0 !important;
}

.browser-box .el-tabs {
  @apply flex flex-col h-full;
}

.browser-box .el-tabs__content {
  @apply overflow-y-auto flex-1;
}

.browser-box .el-tabs__nav {
  @apply space-x-0 !important;
}

.browser-box .el-tabs__item {
  @apply p-1 !important;
}

.browser-box .el-tabs__item:nth-child(4) {
  @apply w-44 !important;
}

.browser-fade-enter-active,
.browser-fade-leave-active {
  transition: all 0.5s;
}

.browser-fade-enter-from, .browser-fade-leave-to /* .fade-leave-active below version 2.1.8 */
{
  transform: translateX(-100%);
  width: 0 !important;
  padding: 0;
}
.browser-fade-enter-active .browser-box,
.browser-fade-leave-active .browser-box{
  transition: all 0.5s;

}
.browser-fade-enter-from .browser-box, .browser-fade-leave-to .browser-box{
  width: 0 !important;
  padding: 0;
}
</style>
<script lang="ts" setup>
import { FileBrowser, ImageBrowser } from '/@/components'
import { nextTick, onMounted, ref, watch } from 'vue'
import { useProject } from '/@/store/project'
import { getPos } from '/@/components/videomain/src/timeline'
import { bus } from '/@/utils/busUtils'
import { CHANGE_VIDEO_PADDING_BOTTOM } from '/@/constants'

const project = useProject()
const setBrower = () => {
  project.changeIsBrowserShow(!project.isBrowserShow)
}
const state = ref('')
const activeName = ref('img')

interface LinkItem {
  value: string
  link: string
}

watch(activeName, function () {
  state.value = ''
})

const links = ref<LinkItem[]>([])

const loadAll = () => {
  return [{ value: '篮球数据集', link: '' }]
}

let timeout: NodeJS.Timeout
const querySearchAsync = (queryString: string, cb: (arg: any) => void) => {
  const results = queryString
    ? links.value.filter(createFilter(queryString))
    : links.value

  clearTimeout(timeout)
  timeout = setTimeout(() => {
    cb(results)
  }, 3000 * Math.random())
}
const createFilter = (queryString: string) => {
  return (restaurant: LinkItem) => {
    return (
      restaurant.value.toLowerCase().indexOf(queryString.toLowerCase()) === 0
    )
  }
}

const handleSelect = (item: LinkItem) => {
  console.log(item)
}

onMounted(() => {
  links.value = loadAll()
})

let dragging = false
const boxWidth = ref(300)
function down () {
  console.log('down')
  dragging = true
  document.body.addEventListener('mousemove', move)

  document.body.addEventListener('mouseup', up)
}
const browserBox = ref(null)
function move (event) {
  if (dragging) {
    const pos = getPos(event, browserBox.value, false)
    const width = Math.min(600, Math.max(300, pos.x))
    console.log(width)
    boxWidth.value = width
  }
}

function up () {
  dragging = false
  document.body.removeEventListener('mousemove', move)
  document.body.removeEventListener('mouseup', up)
}
</script>
