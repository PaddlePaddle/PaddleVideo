<template>
  <div :class="mode_.container + ' ' + theme_+ ' ' + customClass.container">
    <div v-if="cancelable" class="flex justify-end w-full">
      <span class="iconfont icon-guanbi text-indigo-600 hover:text-blue-700"></span>
    </div>
    <p :class="mode_.title">
      <slot name="title">
      </slot>
    </p>
    <div :class="mode_.body">
      <slot name="body">
        <img :src="imgSrc" class="rounded-md">
      </slot>
    </div>

  </div>
</template>

<script setup lang="ts">

const props = defineProps({
  cancelable: { type: Boolean, default: false },
  mode: { type: String, default: 'popup' },
  customClass: {
    type: Object,
    default: {
      container: ''
    }
  },
  imgSrc: String,
  theme: { type: String, default: 'morphy' }
})

const Theme = {
  morphy: 'bg-white shadow',
  glass: 'bg-gray-50 bg-opacity-75 rounded-2xl text-gray-500'
}
// const  = glass
const Mode = {
  info: {
    container: 'inline-flex flex-col space-y-2 items-start justify-center p-2.5 rounded-xl min-w-fit absoulute browser-video-popup',
    title: ' font-bold leading-none text-gray-900 text-sm',
    body: 'space-x-0.5  items-center leading-3 text-gray-500 text-sm space-y-1 text-left'
  },
  massage: {
    container: 'inline-flex flex-col space-y-2.5 items-start justify-start p-3 rounded-2xl min-w-fit',
    title: 'items-start justify-start font-extrabold text-3xl text-gray-600',
    body: 'text-lg text-gray-500 text-left flex-col space-y-3 items-start justify-end flex-1 h-full pt-6 pb-5'
  },
  popup: {
    container: 'inline-flex  items-center justify-center py-2 rounded-xl min-w-fit px-3',
    title: '',
    body: 'flex-1 h-full rounded-lg'
  },
  preview: {
    container: 'inline-flex items-center justify-center p-1 rounded-lg min-w-fit',
    title: '',
    body: 'flex-1 h-full rounded-lg'
  }
}
const theme_ = Theme[props.theme]
const mode_ = Mode[props.mode]
</script>

<style>
.browser-video-popup{
  left: 268px;
  z-index: 99999;
}
</style>
