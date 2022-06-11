<template>
  <div class="container mx-auto">
    <div class="flex pt-5">
      <div class="shadow-lg p-3 flex-1 border border-gray-100">
        <my-video ref="video" :src="require('../../assets/video/2021.mp4')" :tagList="tagList"
                  @addTag="addTag"></my-video>
      </div>
      <div class="ml-3 shadow-lg p-3 border border-gray-100 w-2/6 flex flex-col">
        <div class="tag-list-box flex-1">
          <div v-for="(item, index) in tagList" :key="item.uuid"
               class="tag-box shadow-lg border mb-2 p-2 flex cursor-pointer" @click="changeTime(item.startTime)">
            <div class="dot-box flex items-center">
              <div :style="{ background: item.color }" class="dot w-3 h-3 mr-2 rounded-full"></div>
            </div>
            <div class="flex-1">
              <div class="name text-lg font-bold">{{ item.label }}</div>
              <div class="time text-sm text-gray-400">{{ parseInt(item.startTime) }} - {{
                  parseInt(item.endTime)
                }}
              </div>
            </div>
            <div class="number flex items-center font-bold pr-1">{{ index + 1 }}</div>
          </div>
        </div>
        <div class="button-box">
          <div class="button w-full p-2 flex items-center justify-center bg-blue-500 cursor-pointer text-white"
               @click="exportJson">导出
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import MyVideo from '@/components/MyVideo/Index.vue'

export default {
  components: {
    MyVideo
  },
  data() {
    return {
      tagList: []
    }
  },
  mounted() {
  },
  methods: {
    addTag(item) {
      this.tagList.push(item)
    },
    changeTime(time) {
      this.$refs.video.changeTime(time)
    },
    exportJson() {
      let name = 'basket.json'
      let data = JSON.stringify(
          this.tagList.map((item) => {
            return {
              label_ids: [item.id],
              label_names: [item.label],
              start_id: item.startTime,
              end_id: item.endTime
            }
          })
      )
      let urlObject = window.URL || window.webkitURL || window
      let export_blob = new Blob([data])
      let createA = document.createElement('a')
      createA.href = urlObject.createObjectURL(export_blob)
      createA.download = name
      createA.click()
    }
  }
}
</script>

<style lang="less" scoped></style>
