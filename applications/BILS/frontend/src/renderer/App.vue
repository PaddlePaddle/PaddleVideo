<template>
  <div class="app">
    <router-view></router-view>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useProject } from '/@/store'

const project = useProject()
onMounted(() => {
  project.resetCurVideo()
})
</script>

<style>
.app{
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}
.el-tree-node:focus>.el-tree-node__content {
  background: transparent !important;
}
.light-off .tool-box{
  color: rgba(255, 255, 255, .8);
  z-index: 999999;
  position: relative;
  background: transparent;
}

.light-off .panel-control-button{
  color: #fff;
  z-index: 99;
}
.app.web-full-screen{
  font-size: 0;
}
.web-full-screen .browser-box{
  display: none;
}
.web-full-screen .panel-wrap{
  display: none;
}
.web-full-screen .wrap-box > .icon-zhankai {
  display: none;
}
.web-full-screen .video-wrapper .d-player-wrap.web-full-screen{
  width: 100% !important;
}
</style>
