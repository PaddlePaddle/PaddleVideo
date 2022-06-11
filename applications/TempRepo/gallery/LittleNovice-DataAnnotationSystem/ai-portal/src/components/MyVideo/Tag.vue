<template>
  <div :class="{ 'bg-blue-400': isSelect }" class="tag shadow-md p-2 border m-1 cursor-pointer flex items-center"
       @click="selectTag">
    <div :style="{ background: item.color }" class="dot w-3 h-3 mr-2 rounded-full"></div>
    {{ item.label }}
  </div>
</template>

<script>
export default {
  props: {
    item: {
      type: Object,
      default: null
    },
    isSelect: {
      type: Boolean,
      default: false
    }
  },
  methods: {
    selectTag() {
      this.$emit('selectTag', this.item)
    }
  }
}
</script>
