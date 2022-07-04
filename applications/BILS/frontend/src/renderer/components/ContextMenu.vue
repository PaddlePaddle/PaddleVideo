<template>
  <teleport to="#app">
    <div
      class="context_menu_box z-10"
      :style="style"
      v-show="show"
      @mousedown.stop
      @contextmenu.prevent
    >
      <slot></slot>
    </div>
  </teleport>
</template>

<script>
export default {
  name: 'context-menu',
  emits: ['changeContextMenuShow'],
  data() {
    return {
      triggerHideFn: () => {},
      x: null,
      y: null,
      style: {}
    }
  },
  props: {
    show: Boolean,
    offset: {
      type: Object,
      default: () => {
        return {
          x: 0,
          y: 0
        }
      }
    }
  },
  watch: {
    show(show) {
      if (show) {
        this.bindHideEvents()
      } else {
        this.unbindHideEvents()
      }
    }
  },
  methods: {
    // 绑定隐藏菜单事件
    bindHideEvents() {
      this.triggerHideFn = this.clickDocumentHandler.bind(this)
      document.addEventListener('mousedown', this.triggerHideFn)
      document.addEventListener('mousewheel', this.triggerHideFn)
    },

    // 取消绑定隐藏菜单事件
    unbindHideEvents() {
      document.removeEventListener('mousedown', this.triggerHideFn)
      document.removeEventListener('mousewheel', this.triggerHideFn)
    },

    // 鼠标按压事件处理器
    clickDocumentHandler() {
      this.$emit('changeContextMenuShow', false)
    },

    // 右键事件事件处理
    contextMenuHandler(e) {
      // console.log(e)
      this.x = e.clientX + this.offset.x
      this.y = e.clientY + this.offset.y
      this.layout()
      // e.preventDefault()
    },

    // 布局
    layout() {
      this.style = {
        left: this.x + 'px',
        top: this.y + 'px'
      }
    }
  }
}
</script>

<style lang="scss">
.context_menu_box {
  @apply rounded-md hover:text-red-400;
  position: fixed;
  border: 1px solid #eee;
  background: #fff;
  font-size: 12px;
  .context-menu-item {
    cursor: pointer;
    padding: 3px 5px;
  }
}
</style>
