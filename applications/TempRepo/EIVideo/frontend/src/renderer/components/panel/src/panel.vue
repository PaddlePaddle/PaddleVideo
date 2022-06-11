<template>
  <span class="hover:text-indigo-600 font-bold iconfont icon-zhankai panel-control-button  cursor-pointer pr-2 py-2" @click="setPanel"></span>
  <div class='panel-wrap'>
    <transition name='panel-fade'>
      <div ref='panelBox' class="panel px-4 py-2 bg-white shadow overflow-y-auto rounded-2xl h-screen" :style='{width: boxWidth + "px"}' v-show='project.isPanelShow'>
        <el-tabs v-model="activeName" class="demo-tabs" @tab-click="handleClick">
          <el-tab-pane disabled>
            <template #label>
            </template>
          </el-tab-pane>
          <el-tab-pane label="标注" name="label">
            <item/>
          </el-tab-pane>
          <el-tab-pane label="设置" name="Config">
            <project-form/>
          </el-tab-pane>
<!--          <el-tab-pane label="模型" name="model">模型</el-tab-pane>-->
        </el-tabs>
        <div class='border-left-box'  @mousedown='down'></div>
      </div>
    </transition>
  </div>
</template>
<script lang="ts" setup>
import { Ref, ref } from 'vue'
import { Item, projectForm } from '/@/components/content'
import { useProject } from '/@/store/project'
import { getPos } from '/@/components/videomain/src/timeline'
const project = useProject()
const setPanel = () => {
  project.changeIsPanelShow(!project.isPanelShow)
}
const activeName = ref('label')

const panelBox: Ref<HTMLElement | null> = ref(null)
const handleClick = (tab: string, event: Event) => {
  // console.log(tab, event)
}

let dragging = false
const defaultWidth = 320
const boxWidth = ref(320)
function down () {
  console.log('down')
  dragging = true
  document.body.addEventListener('mousemove', move)

  document.body.addEventListener('mouseup', up)
}
function move (event) {
  if (dragging) {
    const pos = getPos(event, panelBox.value, false)
    const width = Math.min(620, Math.max(320, boxWidth.value - pos.x))
    console.log(pos)
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
<style>
.border-left-box{
  height: 100%;
  width: 2px;
  background: #eee;
  cursor: e-resize;
  position: absolute;
  left: 0;
  top: 0;
}
.panel-fade-enter-active, .panel-fade-leave-active{
  transition: all .5s;
}
.panel-fade-enter-from, .panel-fade-leave-to {
  transform: translateX(100%);
  width: 0 !important;
  padding: 0;
}
.panel-wrap{
  min-width: 1px;
  height: 100vh;
  overflow: hidden;
}
</style>
