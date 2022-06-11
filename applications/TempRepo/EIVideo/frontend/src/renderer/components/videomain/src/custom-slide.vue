<template>
  <div
    class="custom-slide-box"
    ref="customSlideBox"
    @mousedown.stop="mouseDownHandle"
  >
    <div class="inner-box" @mousemove="mousemoveHandle">
      <div class="custom-slide-bar" :style="{ width: width + '%' }"></div>
      <div class="d-slider__cursor" :style="{ left: hoverLeft * 100 + '%' }">
        <div
          class="d-slider__tips"
          ref="refTips"
          :style="{ left: hoverTipsLeft }"
          v-show="hoverText"
        >
          {{ hoverText }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { defineProps, onMounted, ref, watch } from 'vue'
const props = defineProps({
  duration: null,
  realCurrentTime: null
})

const width = ref(0)
const refTips = ref()
watch(props.realCurrentTime, function () {
  if (props.realCurrentTime.value == null || props.duration.value == null) {
    width.value = 0
  }
  width.value = (props.realCurrentTime.value / props.duration.value) * 100
})

const customSlideBox = ref()
function setCurrentTime(event: any) {
  const newCurrentTime = getPosition(event) * props.duration.value
  ;(document.querySelector('#dPlayerVideoMain') as any).currentTime =
    newCurrentTime
}

const dragging = ref(false)

const hoverText = ref()

function mouseDownHandle(event: any) {
  event.preventDefault()
  dragging.value = true
  setCurrentTime(event) // 设置当前位置
  width.value = getPosition(event) * 100
  document.addEventListener('mousemove', onDragging)
  document.addEventListener('touchmove', onDragging)
  document.addEventListener('mouseup', onDragEnd)
  document.addEventListener('touchend', onDragEnd)
}

function onDragging(event: any) {
  width.value = getPosition(event) * 100
  setCurrentTime(event) // 设置当前位置
}

function onDragEnd() {
  if (dragging.value) {
    document.removeEventListener('mousemove', onDragging)
    document.removeEventListener('touchmove', onDragging)
    document.removeEventListener('mouseup', onDragEnd)
    document.removeEventListener('touchend', onDragEnd)
    document.removeEventListener('contextmenu', onDragEnd)
    setTimeout(() => {
      dragging.value = false
    }, 0)
  }
}
const hoverTipsLeft = ref()
const hoverLeft = ref()
function mousemoveHandle(event: any) {
  hoverLeft.value = getPosition(event)
  hoverText.value = timeFormat(getPosition(event) * props.duration.value)
  const refSliderEl = customSlideBox.value as HTMLButtonElement
  // 提示宽的一半宽度
  const refTipsWidth = (refTips.value as HTMLButtonElement).clientWidth / 2
  const movePositon = event.clientX - refSliderEl.getBoundingClientRect().left
  // 如果当前往左的偏移量大于提示框宽度
  if (movePositon < refTipsWidth) {
    hoverTipsLeft.value = refTipsWidth - movePositon + 'px'
  } else if (refSliderEl.clientWidth - movePositon < refTipsWidth) {
    // 如果当前往右的偏移量大于提示框宽度  （总宽度-当前移动位置）< tips一半的宽度
    hoverTipsLeft.value =
      refSliderEl.clientWidth - movePositon - refTipsWidth + 'px'
  } else {
    hoverTipsLeft.value = '50%'
  }
}

const getPosition = (ev: any) => {
  const refSliderEl = customSlideBox.value as HTMLButtonElement
  let value = 0
  value =
    (ev.clientX - refSliderEl.getBoundingClientRect().left) /
    refSliderEl.clientWidth
  return value < 0 ? 0 : value > 1 ? 1 : value
}

const timeFormat = (time: any) => {
  // console.log(time)
  let hh: any = ~~(time / 3600)
  let mm: any = ~~((time % 3600) / 60)
  let ss: any = ~~(time % 60) // 取整
  hh = hh < 10 ? '0' + hh : hh // 个位数补0
  mm = mm < 10 ? '0' + mm : mm // 个位数补0
  ss = ss < 10 ? '0' + ss : ss // 个位数补0
  return `${hh}:${mm}:${ss}`
}
</script>

<style lang="scss" scoped>
.custom-slide-box {
  background: #717171;
  height: 5px;
  position: relative;
  cursor: pointer;
  .inner-box {
    width: 100%;
    height: 100%;
  }
  &:hover {
    .custom-slide-bar {
      &:before {
        transform: translateY(-50%) scale(1);
      }
    }
  }
  .custom-slide-bar {
    position: absolute;
    height: 100%;
    background: rgba(108, 114, 127, 0.8) !important;
    left: 0;
    &:before {
      transform: translateY(-50%) scale(0);
      display: block;
      content: '';
      position: absolute;
      right: -6px;
      top: 50%;
      width: 12px;
      height: 12px;
      transition: 0.2s;
      border-radius: 50%;
      background: #fff;
    }
  }
  &:hover {
    .d-slider__cursor {
      opacity: 1;
      display: block;
    }
  }
  .d-slider__cursor {
    position: absolute;
    top: 0;
    width: 1px;
    display: none;
    opacity: 0;
    .d-slider__tips {
      padding: 4px;
      box-sizing: border-box;
      display: block;
      font-size: 12px;
      background: rgba(0, 0, 0, 0.6);
      border-radius: 3px;
      transform: translate(-50%);
      color: #fff;
      position: absolute;
      white-space: nowrap;
      z-index: 2;
      bottom: 14px;
    }
  }
}
</style>
