<script setup lang="ts">
import {getPos} from './timeline'
import {computed, nextTick, ref, watch} from 'vue'
import Linelabel from './label.vue'
import {VideoTrack} from '/@/components'
import {useLabelsStore, useProject} from '/@/store'
import {bus} from '/@/utils/busUtils'
import {CHANGE_PLAY_CURRENT_TIME, CHANGE_VIDEO_PADDING_BOTTOM} from '/@/constants'

const Labels = useLabelsStore()
const { setdotsPos, changeSegment, getLabels, getTime } = Labels
const labels = computed(() => getLabels())

const slider = ref()
const newDots = true
const getdotsPos = (e) => {
  const pos = getPos(e, slider.value, false).x
  const w = slider.value.getBoundingClientRect().width
  return ((pos / w) * 100).toFixed(2)
}
const addSegment = (e) => {
  setdotsPos( getdotsPos(e) )
}
const props = defineProps({
  videotrack: String,
  trackslen: String,
  min: Number,
  max: Number,
  duration: Number,
  realCurrentTime: Number
})

const width = ref(0)

const project = useProject()
watch(props.realCurrentTime, function () {
  if ((project.videoMeta(project.curVideo as string) as any) == null) {
    return
  }
  const duration = (project.videoMeta(project.curVideo as string) as any).duration
  if (props.realCurrentTime.value == null || duration == null) {
    width.value = 0
  }
  width.value = (props.realCurrentTime.value / duration) * 100
})

const scaleArr = ['100%', '125%', '150%', '175%', '200%', '225%', '250%', '275%', '300%', '350%', '400', '450%', '500%', '600%', '700%', '800%', '900%', '1000%']

const scaleIndex = ref(0)

const nowScale = computed(() => {
  return scaleArr[scaleIndex.value]
})

function wheel (event) {
  if (event.ctrlKey) {
    if (event.deltaY > 0) {
      // 缩小
      if (scaleIndex.value > 0) {
        scaleIndex.value--
      }
    } else {
      // 放大
      if (scaleIndex.value < scaleArr.length - 2) {
        scaleIndex.value++
      }
    }
  }
}

let dragging = false
const wrapperHeight = ref(65)
function down () {
  dragging = true
  document.body.addEventListener('mousemove', move)
  document.body.addEventListener('mouseup', up)
}

function move (event) {
  if (dragging) {
    const pos = getPos(event, slider.value, false)
    const height = Math.min(130, Math.max(65, pos.y))
    bus.emit(CHANGE_VIDEO_PADDING_BOTTOM, wrapperHeight.value)
    nextTick(() => {
      wrapperHeight.value = height
    })
  }
}

bus.on('CHANGE_PIC_BOX_HEIGHT', function () {
  wrapperHeight.value = 65
  bus.emit(CHANGE_VIDEO_PADDING_BOTTOM, 65)
})

function up () {
  dragging = false
  document.body.removeEventListener('mousemove', move)
  document.body.removeEventListener('mouseup', up)
}
const mousemove = (e) => {
  const v = getdotsPos(e)
  bus.emit(
    CHANGE_PLAY_CURRENT_TIME,
    getTime(v)
  )
}
</script>

<template>
  <div class="custom-slider-picture-wrapper" :style='{height: wrapperHeight + "px"}' @wheel='wheel'>
    <div class='custom-slider-picture-inner' :style='{width: nowScale}'>
      <div
        ref="slider"
        class="inline-flex custom-slider-box bg-yellow-200 w-full h-fit rounded ring-1 ring-inset ring-black ring-opacity-0 relative"
        id="slider.value"
        @click.alt="addSegment"
        @mousemove="mousemove"
      >
        <div
          class="label"
          :style="{ height: trackslen }"
          v-for="(item, index) in labels"
          :uuid="item.uuid"
          :key="item.uuid"
        >
          <linelabel
            v-model:item="labels[index]"
            :slider="slider"
            @lineMove="changeSegment"
          ></linelabel>
        </div>
      </div>
      <VideoTrack :videotrack="props.videotrack"></VideoTrack>
      <div class="process-box" :style="{ width: width + '%' }"></div>
    </div>
    <div class='border-bottom' @mousedown='down'></div>
  </div>
</template>

<style>
.custom-slider-picture-wrapper {
  position: fixed;
  overflow: hidden;
  left: 0;
  right: 0;
  bottom: 0;
  font-size: 0;
  background: rgb(249 250 251 / var(--tw-bg-opacity)) !important;
  overflow-x: auto;
  width: 100%;
  cursor: s-resize;
}
.custom-slider-picture-wrapper .border-bottom{
  height: 2px;
  width: 100%;
  cursor: s-resize;
  position: absolute;
  bottom: 0;
}
.custom-slider-picture-inner{
  height: 100%;
  position: relative;
}
.custom-slider-box {
  padding: 0;
  height: 100%;
  position: absolute;
  left: 0;
  right: 0;
  background: transparent !important;
}
.process-box {
  position: absolute;
  top: 0;
  bottom: 0;
  background: rgba(117, 175, 237, 0.38) !important;
  pointer-events: none;
  z-index: 99;
  border-right: 1px solid #fff;
  box-sizing: content-box;
}
</style>
