<template>
  <!--  <div class="mx-10 h-screen w-screen bg-yellow justify-center relative overflow-hidden sm:py-12">-->
  <div class="flex flex-row space-x-1.5 items-center ">
    <span class="iconfont icon-rewind" @click="project.loadVideo(project.project.curVideo, -1)"></span>
    <span class="iconfont icon-play before:text-xl" v-if="play === 'pause'"></span>
    <span class="icon iconfont icon-zanting  before:text-xl" v-else-if="play === 'play'"></span>
    <span class="icon iconfont icon-zhongbo-m  before:text-xl" v-else-if="play === 'ended'"></span>
    <span class="iconfont icon-fast-forward" @click="project.loadVideo(project.project.curVideo, 1)"></span>
    <div class="control-setting-button iconfont icon-more before:font-bold ">
      <div class="control-setting-pop"></div>
      <div class="dom-tool-enhance"></div>
    </div>
  </div>
  <!--  </div>-->
</template>

<style lang="scss">
.control-setting-button {
  position: relative;

  &:hover {
    .control-setting-box {
      display: block;
      opacity: 1;
    }
  }

  .dom-tool-enhance {
    left: -50%;
  }
}
</style>
<script lang="ts" setup>
import {useProject} from '/@/store/project'
import {bus} from "/@/utils/busUtils";
import {PAUSE_VIDEO} from "/@/constants";
import {ref} from "vue";

const project = useProject()
const play = ref('play')
bus.on(PAUSE_VIDEO, function (e) {
  play.value = e
})
</script>
