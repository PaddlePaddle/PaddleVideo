<script setup lang="ts">
import borderline from './borderline.vue'
import contextmenu from '/@/components/ContextMenu.vue'
import { useLabelsStore, useProject } from '/@/store'
import { bus } from '/@/utils/busUtils'
import {
  CHANGE_AUTO_HIDE_CONTROL, CHANGE_DOUBLE_CLICK_INDEX,
  CHANGE_PLAY_CURRENT_TIME,
  PLAY_SEGMENT
} from '/@/constants'
import { ref } from 'vue'
const Labels = useLabelsStore()
const { getSegment, deleteLabelById } = Labels
const isContextMenuShow = ref(false)
const props = defineProps({
  item: Object,
  slider: HTMLElement
})
const emits = defineEmits(['lineMove'])

const project = useProject()
function getRealLeft(left) {
  if (project.curVideo == null || project.curVideo === '') {
    return null
  }
  const duration = (project.videoMeta(project.curVideo as string) as any)
    .duration
  const scale = ((left / duration) * 10000) / 100
  return scale
}

function handlerClick(uuid) {
  getSegment(uuid)
  bus.emit(
    CHANGE_PLAY_CURRENT_TIME,
    Math.min(props.item.range[0], props.item.range[1])
  )
}
function handlerDoubleClick() {
  // console.log('handlerDoubleClick============================')
  props.item.edit = true
  bus.emit(PLAY_SEGMENT, [
    Math.min(props.item.range[0], props.item.range[1]),
    Math.max(props.item.range[0], props.item.range[1])
  ])
}

const contextMenu = ref()

function showContextMenu(event) {
  contextMenu.value.contextMenuHandler(event)
  isContextMenuShow.value = true
}

function deleteSelectLabel(uuid) {
  isContextMenuShow.value = false
  deleteLabelById(uuid)
}

function changeContextMenuShow(flag) {
  isContextMenuShow.value = flag
}
</script>
<template>
  <borderline
    :left="getRealLeft(item.range[0])"
    :slider="slider"
    @changeBorder="(e) => emits('lineMove', e, 0, item.uuid)"
  ></borderline>
  <div
    @contextmenu="showContextMenu"
    @dblclick="handlerDoubleClick"
    class="process h-full opacity-70 absolute"
    :style="{
      background: item.color,
      left:
        Math.min(getRealLeft(item.range[0]), getRealLeft(item.range[1])) + '%',
      backgroundColor: item.color,
      width:
        Math.abs(getRealLeft(item.range[0]) - getRealLeft(item.range[1])) + '%'
    }"
    @click="handlerClick(item.uuid)"
  ></div>
  <borderline
    :left="getRealLeft(item.range[1])"
    :slider="slider"
    @changeBorder="(e) => emits('lineMove', e, 1, item.uuid)"
  ></borderline>
  <contextmenu
    ref="contextMenu"
    :offset="{ x: 0, y: 15 }"
    :show="isContextMenuShow"
    @changeContextMenuShow="changeContextMenuShow"
  >
    <div class="context-menu-item" @click="deleteSelectLabel(item.uuid)">
      删除
    </div>
  </contextmenu>
</template>
