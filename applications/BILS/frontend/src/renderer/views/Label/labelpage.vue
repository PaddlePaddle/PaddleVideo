<template>
    <label-range-card></label-range-card>
  <div class="inline-flex items-start wrap-box">
    <browser></browser>
    <div class='center-wrapper-box flex-1 overflow-hidden inline-flex'>
      <div class="inline-flex relative flex-col overflow-hidden flex-1 h-screen items-start justify-start">
        <tool-box></tool-box>
        <div class='video-out-box' :style='{paddingBottom: paddingBottom + "px"}'>
          <timeline></timeline>
        </div>
        <label-list></label-list>
      </div>
    </div>
    <panel></panel>
  </div>
</template>
<script setup lang="ts">
import { Browser, Panel, Timeline, ToolBox, LabelRangeCard, LabelList } from '/@/components'
import { ref } from 'vue'
import { bus } from '/@/utils/busUtils'
import { CHANGE_VIDEO_PADDING_BOTTOM, CHANGE_VIDEO_PADDING_STATE } from '/@/constants'
const paddingBottom = ref(128)
const oldVal = ref(65)
const isShow = ref(true)
bus.on(CHANGE_VIDEO_PADDING_BOTTOM, (val) => {
  if (val) {
    oldVal.value = val
  }
  if (isShow.value) {
    paddingBottom.value = 64 + val
  } else {
    paddingBottom.value = 64
  }
})

bus.on(CHANGE_VIDEO_PADDING_STATE, (flag) => {
  if (flag) {
    // console.log(oldVal.value)
    isShow.value = true
    paddingBottom.value = 64 + oldVal.value
  } else {
    isShow.value = false
    paddingBottom.value = 64
  }
})
</script>

<style lang="scss" scoped>
.wrap-box {
  width: 100%;
}
.video-out-box{
  transform: scale(1);
  width: 100%;
  z-index: 1;
}
</style>
