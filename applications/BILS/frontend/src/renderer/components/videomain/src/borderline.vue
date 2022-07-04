<script setup lang="ts">
import { getPos } from './timeline'
import { computed, ref, watch } from 'vue'

const props = defineProps({
  left: Number,
  slider: HTMLElement
})
const line = ref()
// 鼠标控制移动
let dragging = false // 控制移动状态
// 键盘方向键控制移动
const emits = defineEmits(['changeBorder'])

// 鼠标按下后的函数,e为事件对象
function down(e) {
  dragging = true
  document.body.addEventListener('mousemove', move)
  document.body.addEventListener('mouseup', up)
}

// 鼠标移动调用的函数
function move(e) {
  if (dragging) {
    const pos = getPos(e, props.slider, false).x
    const w = props.slider.getBoundingClientRect().width
    const v = ((pos / w) * 100).toFixed(2)
    emits('changeBorder', Math.min(Math.max(0, parseFloat(v)), 100))
  }
}

watch(() => props.left, () => {
  line.value.style.left = props.left + '%'
})

// 释放鼠标的函数
function up(e) {
  dragging = false
  document.body.removeEventListener('mousemove', move)
  document.body.removeEventListener('mouseup', up)
}

function keyDown(e) {
  e = e || window.event // 兼容IE
  const code = e.keyCode
  if (code === 37) onLeft()
  if (code === 39) onRight()
}

function onLeft() {
  if (props.left <= 0) emits('changeBorder', 0)
  else emits('changeBorder', (parseFloat(props.left) - 1).toFixed(2))
}

function onRight() {
  if (props.left >= 100) emits('changeBorder', 100)
  else {
    emits('changeBorder', (parseFloat(props.left) + 1).toFixed(2))
  }
}

function selectBorder() {
  // console.log(window.keydownEvent)
  if (window.keydownEvent) {
    document.body.removeEventListener('keydown', window.keydownEvent)
  }
  window.keydownEvent = keyDown
  document.body.addEventListener('keydown', window.keydownEvent)
}

</script>

<template>
  <div ref="line" class="wrap-box h-full w-0.5 bg-white absolute z-10 cursor-e-resize" :style="{left: left+'%'}"
       @click="selectBorder" @mousedown.native="down"></div>
</template>
