<template>
  <div class="video-wrapper top-0" @mouseover='mouseOver' @mouseenter='mouseEnter'>
    <vue3VideoPlay
      :class="{ test: true }"
      ref="videoRef"
      :color="color"
      poster=""
      v-bind="options"
      @lightOffChange="lightOffChange"
      @timeupdate="timeUpdate"
      @durationchange="durationchange"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, createApp, h, nextTick, onMounted, onUnmounted, reactive, Ref, ref, toRefs, watch } from 'vue'
import ControlSetting from './control-setting.vue'
import { Control, LineLabel } from '/@/components'
import { bus } from '/@/utils/busUtils'
import {
  PLAY_VIDEO,
  CHANGE_AUTO_HIDE_CONTROL,
  CHANGE_HIDE_PIC_BOX,
  CHANGE_PLAY_CURRENT_TIME,
  PLAY_SEGMENT,
  PAUSE_VIDEO,
  CHANGE_VIDEO_PADDING_STATE, CHANGE_MENU_POPUP_SHOW
} from '/@/constants/busConstants'
import { useService } from '/@/composables'
import customSlide from './custom-slide.vue'
import { useProject } from '/@/store/project'
import {
  PACKAGE_FOLDER,
  TIMELINE_THUMBNAIL_FOLDER
} from '/@/constants/videoConstants'
import { useLabelsStore } from '/@/store'
import { storeToRefs } from 'pinia'

const fileService = useService('FileService')
const videoRef: Ref<any> = ref()
const max = ref(0)
const isControlAutoHide = ref(false)
const isHidePic = ref()

const isShowControl = ref(true)

let controlTimeout = null

bus.on(CHANGE_HIDE_PIC_BOX, function (flag) {
  isHidePic.value = flag
  if (props.videotrack.length == 0) {
    return
  }
  if (flag) {
    (document.querySelector('.video-wrapper') as HTMLElement).classList.remove('hide-pic-box')
    bus.emit(CHANGE_VIDEO_PADDING_STATE, true)
  } else {
    (document.querySelector('.video-wrapper') as HTMLElement).classList.add('hide-pic-box')
    bus.emit(CHANGE_VIDEO_PADDING_STATE, false)
  }
})

bus.on(CHANGE_AUTO_HIDE_CONTROL, (flag) => {
  isControlAutoHide.value = flag
  if (flag) {
    (document.querySelector('.video-wrapper') as HTMLElement).classList.remove('show-control')
  } else {
    (document.querySelector('.video-wrapper') as HTMLElement).classList.add('show-control')
  }
})

function mouseOver () {
  if (controlTimeout) {
    clearTimeout(controlTimeout)
  }
  if (isControlAutoHide.value) {
    controlTimeout = setTimeout(function () {
      bus.emit(CHANGE_MENU_POPUP_SHOW, false)
    }, 2500)
  }
}

onUnmounted(function () {
  if (controlTimeout) {
    clearTimeout(controlTimeout)
  }
})

function mouseEnter () {
  if (controlTimeout) {
    clearTimeout(controlTimeout)
  }
  isShowControl.value = true
  bus.emit(CHANGE_MENU_POPUP_SHOW, true)
}

const trackslen = computed(() => {
  console.log(65 * props.videotrack.length + 'px')
  return 65 * props.videotrack.length + 'px'
})
nextTick(() => {
  const Media = document.getElementsByClassName('d-player-video-main')[0]
  Media.addEventListener('durationchange', () => {
    max.value = Media.duration
  })
  // const Tracks = h(LineLabel, {
  //   videotrack: props.videotrack,
  //   trackslen: trackslen,
  //   min: 0,
  //   max: max,
  //   duration: duration,
  //   realCurrentTime: realCurrentTime
  // })
  // const tracks = createApp(Tracks)
  // tracks.mount('.d-slider__preload')
})

const props = reactive({
  videotrack: []
})

watch([props.videotrack], function () {
  console.log('监听到')
  nextTick(() => {
    if (props.videotrack.length != 0) {
      if (isHidePic.value) {
        (document.querySelector('.video-wrapper') as HTMLElement).classList.remove('hide-pic-box')
        bus.emit(CHANGE_VIDEO_PADDING_STATE, true)
      }
    } else {
      (document.querySelector('.video-wrapper') as HTMLElement).classList.add('hide-pic-box')
      bus.emit(CHANGE_VIDEO_PADDING_STATE, false)
      console.log((document.querySelector('.video-wrapper') as HTMLElement).classList)
    }
  })
}, {
  deep: true,
  immediate: true
})

// const linelabel = h(LineLabel, { start: start.value, end: end.value, cur: cur })
nextTick(() => {
  const control = createApp(Control)
  control.mount('.d-tool-time>span:nth-child(2)')
  const controlSetting = createApp(ControlSetting)
  controlSetting.mount('.control-setting-button .control-setting-pop')
})
nextTick(() => {
  const elements = document.querySelectorAll(
    '.d-tool-bar:last-child .d-tool-item'
  )
  for (const element of elements) {
    const div = document.createElement('div')
    div.classList.add('dom-tool-enhance')
    element.appendChild(div)
  }
})

const duration = ref()
const realCurrentTime = ref()

function timeUpdate(event: any) {
  realCurrentTime.value = event.currentTime || event.target.currentTime
}

function durationchange(event: any) {
  duration.value = event.duration || event.target.duration || 0
}

nextTick(() => {
  const slider = createApp(customSlide, {
    duration: duration,
    realCurrentTime: realCurrentTime
  })
  const slideBox = document.createElement('div')
  slideBox.classList.add('custom-slide-wrapper')
  const parent = document.querySelector('.d-control-tool') as HTMLElement
  parent.insertBefore(slideBox, parent.childNodes[1])
  slider.mount('.custom-slide-wrapper')
})

nextTick(() => {
  const lineLabel = createApp(LineLabel, {
    videotrack: props.videotrack,
    duration: duration,
    realCurrentTime: realCurrentTime
  })
  const parent = document.createElement('div')
  parent.classList.add('line-label-box')
  document.querySelector('.d-player-control').appendChild(parent)
  lineLabel.mount('.line-label-box')
})

nextTick(() => {
  console.log('监听事件');
  (document.querySelectorAll('.fullScreen-btn')[0] as HTMLElement).addEventListener('click', () => {
    console.log('change')
    options.webFullScreen = !options.webFullScreen
    if (options.webFullScreen) {
      (document.querySelector('.app') as HTMLElement).classList.add('web-full-screen')
      bus.emit('CHANGE_PIC_BOX_HEIGHT')
    } else {
      (document.querySelector('.app') as HTMLElement).classList.remove('web-full-screen')
    }
  })
})

function hideWebFullScreen(event: any) {
  if (event.keyCode === 27) {
    if (options.webFullScreen) {
      options.webFullScreen = false
      document.exitFullscreen();
      (document.querySelector('.app') as HTMLElement).classList.remove('web-full-screen')
    }
  }
}

onMounted(() => {
  window.addEventListener('keydown', hideWebFullScreen)
})

onUnmounted(() => {
  window.removeEventListener('keydown', hideWebFullScreen)
})

const color = '#4942F5'
const options = reactive({
  width: '800px', // 播放器高度
  height: '450px', // 播放器高度
  color: '#fff', // 主题色
  title: '', // 视频名称
  src: '', // 视频源
  muted: false, // 静音
  webFullScreen: false,
  speedRate: ['0.75', '1.0', '1.25', '1.5', '2.0'], // 播放倍速
  autoPlay: false, // 自动播放
  loop: false, // 循环播放
  mirror: false, // 镜像画面
  ligthOff: true, // 关灯模式
  volume: 0.01, // 默认音量大小
  control: true, // 是否显示控制
  currentTime: 0, // 时间
  controlBtns: [
    'audioTrack',
    'quality',
    'speedRate',
    'volume',
    'setting',
    'pip',
    'pageFullScreen',
    'fullScreen'
  ] // 显示所有按钮,
})

function lightOffChange(value: boolean) {
  if (value) {
    (document.body.querySelector('.app') as HTMLElement).classList.add('light-off');
    (document.body.querySelector('.d-player-lightoff') as HTMLElement) && (document.body.querySelector('.d-player-lightoff') as HTMLElement).remove()
    const lightOffDiv = document.createElement('div')
    lightOffDiv.classList.add('d-player-lightoff');
    (document.body.querySelector('.center-wrapper-box') as HTMLElement).append(lightOffDiv)
  } else {
    (document.body.querySelector('.app') as HTMLElement).classList.remove('light-off');
    (document.body.querySelector('.d-player-lightoff') as HTMLElement).remove()
  }
}

bus.on(PLAY_VIDEO, async function (videoItem: any) {
  // console.log(PLAY_VIDEO)
  options.src = (await fileService.getFilePrefix()) + videoItem.path
  options.currentTime = 0
  setTimelineImg(videoItem)
  if (videoRef.value != null) {
    videoRef.value.play()
  }
})
const Label = useLabelsStore()
const { curLabel } = storeToRefs(Label)
bus.on(PLAY_SEGMENT, async function () {
  videoRef.value.play()
  const Media = document.querySelector('#dPlayerVideoMain') as any
  Media.addEventListener('timeupdate', function fn() {
    if (Media.currentTime > curLabel.value.range[1]) {
      videoRef.value.pause()
      Media.removeEventListener('timeupdate', fn)
    }
  }, false)
})
nextTick(() => {
  const Media = document.querySelector('#dPlayerVideoMain') as any
  Media.addEventListener('pause', function fn() {
    bus.emit(PAUSE_VIDEO, 'pause')
    // videoRef.value.pause()
  }, false)
  Media.addEventListener('play', function fn() {
    bus.emit(PAUSE_VIDEO, 'play')
    // videoRef.value.pause()
  }, false)
  Media.addEventListener('ended', function fn() {
    bus.emit(PAUSE_VIDEO, 'ended')
    // videoRef.value.pause()
  }, false)
})

bus.on(CHANGE_PLAY_CURRENT_TIME, function (time) {
  const Media = document.querySelector('#dPlayerVideoMain') as any
  Media.currentTime = time
})

const projectStore = useProject()

async function setTimelineImg(videoItem: any) {
  props.videotrack.splice(0, props.videotrack.length)
  const arr = await genTimelineImgList(videoItem)
  props.videotrack.push(arr)
}

async function genTimelineImgList(videoItem: any) {
  const arr = []
  for (let i = 0; i < 30; i++) {
    const parentFolderPath =
      projectStore.baseConfig.projectPath +
      (await fileService.getSep()) +
      PACKAGE_FOLDER
    const folderPath =
      parentFolderPath +
      (await fileService.getSep()) +
      TIMELINE_THUMBNAIL_FOLDER +
      (await fileService.getSep()) +
      videoItem.uuid
    arr.push(
      (await fileService.getFilePrefix()) +
      folderPath +
      (await fileService.getSep()) +
      (i + 1) +
      '.png'
    )
  }
  return arr
}

</script>
<style>
.web-full-screen .d-player-control{
  top: auto;
  bottom: 65px !important;
}
.web-full-screen .hide-pic-box .d-player-control{
  top: auto;
  bottom: 0 !important;
}
.hide-pic-box .custom-slider-picture-wrapper {
  display: none;
}

.custom-slide-wrapper {
  flex: 1;
}

.d-player-lightoff {
  position: fixed;
  left: 0;
  top: 0;
  width: 100vw;
  height: 100vh;
  background-color: #000000e6;
}

.d-player-control {
  /*height: calc(70px + v-bind(trackslen)) !important;*/
  @apply px-1.5 fixed bg-gray-50 bg-opacity-60 rounded-2xl text-gray-500 place-items-center !important;
  bottom: auto !important;
  top: 450px;
}

.d-slider {
  /*@apply h-1 !important;*/
  /*@apply -mt-16 !important;*/
}

/*.d-control-progress {*/
/*  !*height: v-bind(trackslen) !important;*!*/
/*  @apply mt-16  !important;*/
/*}*/

.d-tool-item-main {
  @apply bg-gray-50 bg-opacity-60 rounded-2xl text-gray-500 !important;
  bottom: 55px !important;
}

.d-tool-bar .d-tool-item {
  white-space: nowrap !important;
}

.d-control-tool {
  @apply absolute my-3.5 bg-transparent top-0 h-min  !important;
  margin: 0 !important;
  padding: 14px 0 !important;
  background: rgb(249 250 251 / var(--tw-bg-opacity)) !important;
}

.d-tool-bar {
  @apply place-items-center  !important;
}

.d-player-control {
  @apply transform-none  !important;
}

.volume-box {
  padding: 6px 0 !important;
}

.volume-main {
  @apply overflow-hidden  !important;
}

.volume-main .d-slider {
  @apply h-full m-0 !important;
}

.volume-main .d-slider .d-slider__runway {
  height: 100% !important;
}

.volume-main .d-slider .d-slider__runway .d-slider__bar {
  background: rgba(108, 114, 127, 0.8) !important;
  /*background: linear-gradient(to right,#52a0fd 0%,#00e2fa 80%,#00e2fa 100%) !important;*/
}

.volume-main .d-slider .d-slider__runway {
  background: rgba(0, 0, 0, 0.1) !important;
  /*background: linear-gradient(to right,#52a0fd 0%,#00e2fa 80%,#00e2fa 100%) !important;*/
}

.d-tool-bar:nth-child(1) {
  @apply relative flex pl-8 !important;
}

.d-tool-time {
  @apply relative text-gray-500 pl-12 flex space-x-2 !important;
}

.d-tool-time > span:nth-child(1) {
  @apply text-gray-700 !important;
}

.d-tool-time > span:nth-child(2) {
  @apply absolute -left-14 !important;
}

.d-tool-item .d-icon.icon-replay {
  @apply z-20 opacity-0;
}

.d-tool-item .d-icon.icon-pause {
  @apply z-20 opacity-0;
}

.d-tool-item .d-icon.icon-play {
  @apply z-20  opacity-0;
}

.total-time {
  @apply text-gray-500 !important;
}

.d-player-wrap {
  @apply w-full relative !important;
}

.d-player-wrap-hover .d-player-control {
  @apply block opacity-100;
}

.d-control-progress {
  @apply hidden !important;
}

.d-player-wrap .d-player-control .d-control-tool .speed-main li {
  @apply text-gray-500 !important;
}

.custom-slide-bar {
  background: rgba(255, 255, 255, 0.5) !important;
}

.inner-box {
  background: rgba(0, 0, 0, 0.1) !important;
  /*@apply h-28 !important;*/
}

.custom-slide-box {
  background: rgba(255, 255, 255, 0.1) !important;
  /*@apply h-28 !important;*/
}

.d-slider__tips {
  @apply bg-gray-50 bg-opacity-75 rounded-md text-gray-500 !important;
}

/*.d-progress-bar {*/
/*  @apply bg-transparent !important;*/
/*}*/

.d-progress-bar {
  display: none;
  height: 100% !important;
}

.d-slider .d-slider__runway .d-slider__bar[data-v-5a794390]:before {
  /*height: v-bind(trackslen) !important;*/
  @apply shadow-none !important;
}

.d-loading {
  @apply invisible !important;
}

.dom-tool-enhance {
  position: absolute;
  height: 35px;
  left: 0;
  right: 0;
  top: -35px;
}

.video-wrapper {
  width: 100%;
}

.video-wrapper.show-control .d-player-control {
  opacity: 1;
}

.d-player-control {
  opacity: 0;
}

.d-slider__cursor {
  @apply pointer-events-none !important;
}

.d-slider__bar:before {
  @apply pointer-events-none !important;
}

.d-slider__bar {
  @apply pointer-events-none !important;
}

.d-slider__tips {
  @apply pointer-events-none !important;
}

.volume-main .d-slider__bar:before {
  display: none !important;
}

/*.d-slider__runway{*/
/*  @apply pointer-events-none  !important;*/
/*}*/
</style>
