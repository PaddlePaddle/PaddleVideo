<template>
<div class="list-box">
  <div class="name" @click="showChildren" v-if="item.type === 'folder'" @contextmenu="showContentMenu($event, item)">{{item.name}}</div>
  <div class="name" @click="showContent" v-else @contextmenu="showContentMenu($event, item)">{{item.name}}</div>
  <div class="children-box" v-show="isChildrenShow">
    <file-item ref="fileItem" v-for="item in childrenList" :key="item.path" :item="item" v-bind="$attrs"></file-item>
  </div>
</div>
</template>

<script>
import { useService } from '/@/composables'
import { defineComponent } from 'vue'
import getFileModel from '/@/model/FileModel'
const { getChildPaths, getSep, readFile, removeFile } = useService('FileService')
export default defineComponent({
  props: {
    item: {
      type: Object
    }
  },
  data () {
    return {
      isChildrenShow: false,
      childrenList: []
    }
  },
  methods: {
    async showChildren () {
      if (this.item.type === 'file') {
        return null
      }
      if (this.childrenList.length !== 0) {
        this.isChildrenShow = !this.isChildrenShow
        return null
      }
      const childrenPaths = await getChildPaths(this.item.path)
      for (const childPath of childrenPaths) {
        const fullPath = this.item.path + await getSep() + childPath
        this.childrenList.push(await getFileModel(fullPath))
      }
      this.isChildrenShow = true
    },
    async showContent () {
      const content = await readFile(this.item.path, 'utf-8')
      console.log(content)
      this.$emit('content', content)
    },
    showContentMenu ($event, item) {
      this.$emit('showContentMenu', $event, item)
    },
    deleteFile (obj) {
      const index = this.childrenList.findIndex(item => {
        return item.path === obj.path
      })
      if (index >= 0) {
        removeFile(this.childrenList[index].path)
        this.childrenList.splice(index, 1)
      } else {
        for (const key in this.$refs.fileItem) {
          const fileItemElement = this.$refs.fileItem[key]
          fileItemElement.deleteFile(obj)
        }
      }
    }
  }
})
</script>

<style lang="scss" scoped>
.list-box{
  .children-box{
    padding-left: 20px;
  }
}
</style>
