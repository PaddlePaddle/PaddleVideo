<template>
  <div class="flex pb-0.5 track-box">
    <div
      class="inline-flex items-center justify-center h-full w-full rounded-lg overflow-hidden"
    >
      <div
        v-for="frame in trackList"
        class="w-10 md:w-16 lg:w-20 h-full bg-cover bg-center flex-1"
      >
        <img :src="frame" />
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { defineProps, onMounted, ref, toRefs, watch } from 'vue'
import { PACKAGE_FOLDER, TIMELINE_THUMBNAIL_FOLDER } from '/@/constants'
import { useProject } from '/@/store'
import { useService } from '/@/composables'
const fileService = useService('FileService')
defineProps({
  videotrack: {
    type: Array,
    default: function () {
      return []
    }
  }
})
const projectStore = useProject()
const trackList = ref([])
onMounted(async () => {
  if (
    projectStore.curVideoItem &&
    projectStore.curVideoItem.isLoadTimeLineThumbnailFinish
  ) {
    trackList.value.splice(0, trackList.value.length)
    const arr = await genTimelineImgList(projectStore.curVideoItem)
    console.log(arr)
    trackList.value.push(...arr)
  } else {
    trackList.value.splice(0, trackList.value.length)
  }
})

watch(
  projectStore,
  async () => {
    if (
      projectStore.curVideoItem &&
      projectStore.curVideoItem.isLoadTimeLineThumbnailFinish
    ) {
      trackList.value.splice(0, trackList.value.length)
      const arr = await genTimelineImgList(projectStore.curVideoItem)
      // console.log(arr)
      trackList.value.push(...arr)
    } else {
      trackList.value.splice(0, trackList.value.length)
    }
  },
  { deep: true }
)

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

<style scoped>
.track-box{
  @apply h-full;
  user-select: none;
}
.track-box img {
  display: block;
  width: 100%;
  @apply h-full;
  object-fit: cover;
}
</style>
