<template>
  <div class="tag-list-box flex mt-2">
    <Tag v-for="item in tagList" :key="item.id" :isSelect="nowTag && item.id === nowTag.id" :item="item"
         v-bind="$attrs"></Tag>
  </div>
</template>

<script>
import Tag from './Tag'

export default {
  props: {
    nowTag: {
      type: Object,
      default: null
    }
  },
  components: {
    Tag
  },
  data() {
    return {
      tagList: [
        {
          color: 'red',
          id: 0,
          label: '背景',
          startTime: null,
          endTime: null
        },
        {
          color: 'blue',
          id: 1,
          label: '回放',
          startTime: null,
          endTime: null
        },
        {
          color: 'pink',
          id: 2,
          label: '进球-三分球',
          startTime: null,
          endTime: null
        },
        {
          color: 'gray',
          id: 3,
          label: '进球-两分球',
          startTime: null,
          endTime: null
        },
        {
          color: 'yellow',
          id: 4,
          label: '进球-扣篮',
          startTime: null,
          endTime: null
        },
        {
          color: 'purple',
          id: 5,
          label: '罚球',
          startTime: null,
          endTime: null
        },
        {
          color: 'orange',
          id: 6,
          label: '跳球',
          startTime: null,
          endTime: null
        }
      ]
    }
  },
  methods: {
    selectTag(item) {
    }
  }
}
</script>

<style lang="less"></style>
