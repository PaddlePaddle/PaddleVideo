<template>
  <div class="controller-box">
    <div ref="progressBox" class="progress-box wfull h-1.5 bg-gray-200 bg-opacity-60 cursor-pointer"
         @click="progressChangeTime">
      <div :style="{ width: (currentTime / totalTime) * 100 + '%' }" class="progress-bar h-1.5 bg-blue-400">
        <div v-for="item in tagList" :key="item.uuid"
             :style="{ background: item.color, width: getTagWidth(item) + 'px', left: getTagLeft(item) + 'px' }"
             class="h-full absolute top-0"></div>
      </div>
    </div>
    <div class="controller-wrap h-10 flex justify-between">
      <div class="button-wrap flex">
        <div v-show="!isPlay"
             class="iconfont icon-bofang text-lg text-white h-full flex items-center p-2 cursor-pointer"
             @click="play"></div>
        <div v-show="isPlay"
             class="iconfont icon-24gf-pause2 text-lg text-white h-full flex items-center p-2 cursor-pointer"
             @click="pause"></div>
        <div v-show="totalTime != null" class="time-box flex items-center text-white pl-2">{{
            getFormatTime(currentTime)
          }}/ {{ getFormatTime(totalTime) }}
        </div>
      </div>
      <div v-if="nowTag" :style="{ background: nowTag.color }"
           class="tag-box p-1 flex items-center m-2 text-sm text-white">{{ nowTag.label }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    isPlay: {
      type: Boolean,
      default: false
    },
    currentTime: {
      type: Number,
      default: 0
    },
    totalTime: {
      type: Number,
      default: null
    },
    nowTag: {
      type: Object,
      default: null
    },
    tagList: {
      type: Array,
      default() {
        return []
      }
    }
  },
  data() {
    return {
      isShow: true
    }
  },
  mounted() {
  },
  methods: {
    play() {
      this.$emit('update:isPlay', true)
    },
    pause() {
      this.$emit('update:isPlay', false)
    },
    getFormatTime(time) {
      let result = ''
      let h = parseInt(time / 3600),
          m = parseInt((time % 3600) / 60),
          s = parseInt((time % 3600) % 60)
      if (h > 0) {
        result += h + ':'
      }
      result += (m < 10 ? '0' + m : m) + ':'
      result += s < 10 ? '0' + s : s
      return result
    },
    progressChangeTime(event) {
      let width = event.clientX - this.getElementLeft(this.$refs.progressBox)
      let point = width / this.$refs.progressBox.clientWidth
      let newTime = this.totalTime * point
      console.log(newTime)
      this.$emit('changeTime', newTime)
    },
    getElementLeft(ele) {
      let left = ele.offsetLeft
      let current = ele.offsetParent
      while (current) {
        left += current.offsetLeft
        current = current.offsetParent
      }
      return left
    },
    getTagWidth(item) {
      return ((item.endTime - item.startTime) / this.totalTime) * this.$refs.progressBox.clientWidth
    },
    getTagLeft(item) {
      return (item.startTime / this.totalTime) * this.$refs.progressBox.clientWidth
    }
  },
  unmounted() {
  }
}
</script>

<style lang="less">
.controller-box {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
}

.controller-wrap {
  background: rgba(0, 0, 0, 0.5);
}

.progress-bar {
  overflow: hidden;
  position: relative;
}
</style>
