<template>
  <div class="video-box relative">
    <video ref="video" :src="src"></video>
    <controller v-model:isPlay="isPlay" :current-time="currentTime" :nowTag="nowTag" :tagList="tagList"
                :total-time="totalTime" @changeTime="changeTime" @play="play"></controller>
  </div>
  <tag-list :nowTag="nowTag" @selectTag="selectTag"></tag-list>
</template>
<script>
import Controller from './Controller'
import TagList from './TagList'
import {v4 as uuidv4} from 'uuid'

export default {
  components: {
    Controller,
    TagList
  },
  emits: ['addTag'],
  props: {
    src: {
      type: String,
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
      isPlay: false,
      currentTime: null,
      totalTime: null,
      nowTag: null
    }
  },
  watch: {
    isPlay(newVal) {
      if (newVal) {
        this.play()
      } else {
        this.pause()
      }
    }
  },

  async mounted() {
    await this.$nextTick()
    this.$refs.video.addEventListener('timeupdate', this.timeUpdate)
    this.$refs.video.addEventListener('canplay', this.canPlay)
    window.addEventListener('keydown', this.keyDown)
  },
  methods: {
    togglePay() {
      this.isPlay = !this.isPlay
    },
    play() {
      this.$refs.video.play()
    },
    pause() {
      this.$refs.video.pause()
    },
    timeUpdate() {
      this.currentTime = this.$refs.video.currentTime
    },
    canPlay() {
      this.totalTime = this.$refs.video.duration
    },
    changeTime(time) {
      this.$refs.video.currentTime = time
    },
    selectTag(item) {
      if (this.nowTag == null) {
        this.nowTag = item
        this.nowTag.startTime = this.currentTime || 0
      } else if (this.nowTag.id === item.id) {
        this.nowTag.endTime = this.currentTime
        if (this.nowTag.endTime === this.nowTag.startTime || this.nowTag.endTime < this.nowTag.startTime) {
          console.log('时间错误')
        } else {
          let newObj = JSON.parse(JSON.stringify(this.nowTag))
          newObj.uuid = uuidv4()
          this.$emit('addTag', newObj)
          this.nowTag = null
        }
      }
      console.log(this.nowTag)
    },
    keyDown(event) {
      console.log(event.keyCode)
      if (event.keyCode == 32) {
        this.togglePay()
      }
      if (event.keyCode == 37) {
        let newTime = this.currentTime > 5 ? this.currentTime - 5 : 0
        this.changeTime(newTime)
      }
      if (event.keyCode == 39) {
        let newTime = this.currentTime < this.totalTime - 5 ? this.currentTime + 5 : this.totalTime
        this.changeTime(newTime)
      }
    }
  },
  unmounted() {
    window.removeEventListener('keydown', this.keyDown)
  }
}
</script>
<style lang="less">
.video-box {
  position: relative;
}
</style>
