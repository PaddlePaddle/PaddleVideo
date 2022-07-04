<template>
  <div
    class="inline-flex w-full flex-row space-y-1.5 items-center justify-start flex-col pt-3 pb-3"
  >
    <div
      v-for="item in filterArr"
      :key="item.uuid"
      v-loading="item.isLoadTimeLineThumbnail"
      element-loading-text="转换缩略图中"
      class=" img-box"
      :class="{'select': project.curVideo == item.uuid}"
    >
      <div class='img-inner-box'>
        <div class='absolute top-0 right-0'>
          <!--      <div class='absolute top-0 right-0'>-->
          <span class="iconfont color-wodeqianbao before:text-3xl float-right" v-if="item.isLabeled"></span>
          <span class="iconfont color-dengdaitishi before:text-3xl float-right" v-else-if="item.isPlayed"></span>
        </div>
        <img
          :src="item.thumbnail"
          class="w-full h-130-px object-cover cursor-pointer"
          @dblclick="project.loadVideo(item.uuid, null)"
        />
        <div class="video-name">{{ item.name }}</div>
        <card
          mode="info"
          :customClass="customClass"
          theme="glass"
          v-if="project.videoMeta(item.uuid)"
        >
          <template #title>
            名称：{{ project.videoMeta(item.uuid)['videoName'] }}
          </template>
          <template #body>
            <p>帧率：{{ project.videoMeta(item.uuid)['fps'] }}</p>
            <p>大小：{{ project.videoMeta(item.uuid)['size'] }}</p>
            <p>分辨率：{{ project.videoMeta(item.uuid)['resolution'] }}</p>
            <p>时长：{{ project.videoMeta(item.uuid)['duration'] }}秒</p>
          </template>
        </card>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import {Card} from '/@/components'
import {computed, defineProps, watch} from 'vue'
import {useProject} from '/@/store/project'
import {useLabelsStore} from '/@/store'
import {storeToRefs} from "pinia";

const project = useProject()
const customClass = {
  container: 'video-info fixed z-20 absolute left-full -top-1/3 hidden'
}
const prop = defineProps({
  filterValue: String
})
const filterArr = computed(function () {
  if (prop.filterValue == null || prop.filterValue === '') {
    return project.videoList
  }
  return project.videoList.filter((item) => {
    return (item.name as string).indexOf(prop.filterValue as string) >= 0
  })
})

const labelStore = useLabelsStore()
const {getLabels} = labelStore
const {curVideo} = storeToRefs(labelStore)
watch(() => getLabels(), () => {
  // console.log('filterArr===========================================', typeof filterArr)
  if (getLabels()) {
    const index = filterArr.value.findIndex((item) => {
      return item.uuid === curVideo.value
    })
    filterArr.value[index].isLabeled = getLabels().length > 0
  }
}, {deep: true})
</script>
<style>
.img-box:hover .video-info {
  @apply block transition delay-700 duration-300 ease-in-out transform translate-y-14;
}

.browser-box {
  @apply overflow-visible  !important;
}
</style>
<style lang="scss" scoped>
.img-box.select {
  border-left: 5px solid rgb(79 70 229 / var(--tw-bg-opacity));
}

.img-box {
  width: 100%;
  position: relative;
  border-left: 5px solid transparent;
  padding-left: 5px;
  padding: 25% 0;

  .img-inner-box {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    top: 0;

    img {
      height: 100%;
    }
  }

  .video-name {
    position: absolute;
    bottom: 0;
    color: #fff;
    background: rgba(0, 0, 0, 0.3);
    width: 100%;
    font-size: 14px;
    padding: 0 10px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    opacity: 0;
  }

  &:hover {
    border-left: 5px solid rgb(79 70 229 / var(--tw-bg-opacity));

    .video-name {
      opacity: 1;
    }
  }
}
</style>
