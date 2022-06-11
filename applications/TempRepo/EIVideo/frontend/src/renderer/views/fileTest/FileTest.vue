<template>
  <div class="button-box">
    <button @click="selectFile">选择文件</button>
    <button @click="selectFileFolder">选择文件夹</button>
    <button  @click="selectFileToGetAllFile">选择文件夹获取所有子文件</button>
    <button  @click="selectFileToGetVideoFile">选择文件夹获取所有视频文件</button>
    <input v-model="outPath">
    <button  @click="writeFile">格式化输出JSON文件</button>
    <button  @click="getClipBoardContent">输出剪贴板内容</button>
  </div>
  <div class="main-box">
    <div class="left-menu">
      <file-item ref="fileItem" v-for="item in fileList" :key="item.path" :item="item" @content="readContent" @showContentMenu="showContentMenuHandler"></file-item>
    </div>
    <div class="content-box">
      {{content}}
    </div>
  </div>
  <context-menu ref="contextMenu" :offset="{x: -35, y: 15}" :show="isContextMenuShow" @update:show="(show) => this.isContextMenuShow = show">
    <div @click="deleteFile">删除</div>
  </context-menu>
</template>

<script>
import { useDialog, useService } from '/@/composables'
import { defineComponent, reactive } from 'vue'
import FileItem from './FileItem.vue'
import getFileModel from '/@/model/FileModel'
import ContextMenu from '/@/components/ContextMenu.vue'
const { writFile, removeFile, getAllFile } = useService('FileService')
const { format } = useService('PrettierService')

export default defineComponent({
  components: {
    FileItem,
    ContextMenu
  },
  data () {
    return {
      isContextMenuShow: false,
      outPath: null,
      content: '',
      menuFileObj: null
    }
  },
  setup () {
    const { showOpenDialog } = useDialog()
    const fileList = reactive([])
    async function selectFile () {
      const data = await showOpenDialog({
        title: '选择文件',
        properties: ['multiSelections', 'openFile']
      })
      this.pathsHandler(data.filePaths)
    }
    async function selectFileFolder () {
      const data = await showOpenDialog({
        title: '选择文件',
        properties: ['multiSelections', 'openDirectory']
      })
      this.pathsHandler(data.filePaths)
    }
    async function selectFileToGetAllFile () {
      const data = await showOpenDialog({
        title: '选择文件',
        properties: ['multiSelections', 'openDirectory']
      })
      this.allFilePathsHandler(data.filePaths)
    }
    async function selectFileToGetVideoFile () {
      const data = await showOpenDialog({
        title: '选择文件',
        properties: ['multiSelections', 'openDirectory']
      })
      const expands = ['mp4', 'avi', 'rmvb']
      this.allFilePathsHandler(data.filePaths, expands)
    }
    return {
      selectFile,
      selectFileFolder,
      selectFileToGetAllFile,
      selectFileToGetVideoFile,
      fileList
    }
  },
  methods: {
    pathsHandler (pathArr) {
      pathArr.forEach(async path => {
        const index = this.fileList.findIndex(item => {
          return path === item.path
        })
        if (index >= 0) {
          return null
        }
        this.fileList.push(await getFileModel(path))
      })
    },
    allFilePathsHandler (pathArr, expands) {
      this.fileList.splice(0)
      pathArr.forEach(async path => {
        try {
          const arr = await getAllFile(path, expands)
          for (const filePath of arr) {
            this.fileList.push(await getFileModel(filePath))
          }
        } catch (err) {
          console.error('出错')
        }
      })
    },
    readContent (content) {
      console.log('readContent')
      this.content = content
    },
    async writeFile () {
      const config = {
        trailingComma: 'es5',
        tabWidth: 4,
        semi: false,
        singleQuote: true,
        parser: 'json'
      }
      const data = await format(this.content, config)
      console.log(data)
      await writFile(this.outPath, data)
    },
    showContentMenuHandler ($event, item) {
      console.log('=============成功拦截===============')
      console.log($event)
      console.log(item)
      this.isContextMenuShow = true
      this.menuFileObj = item
      this.$refs.contextMenu.contextMenuHandler($event)
    },
    deleteFile () {
      const index = this.fileList.findIndex(item => {
        return item.path === this.menuFileObj.path
      })
      if (index >= 0) {
        removeFile(this.fileList[index].path)
        this.fileList.splice(index, 1)
      } else {
        console.log(this.$refs.fileItem)
        for (const key in this.$refs.fileItem) {
          const fileItemElement = this.$refs.fileItem[key]
          fileItemElement.deleteFile(this.menuFileObj)
        }
      }
      this.isContextMenuShow = false
      this.menuFileObj = null
    }
  }
})
</script>

<style lang="scss" scoped>
.button-box{
  button{
    border: 1px solid #ddd;
    padding: 5px;
    margin: 5px;
  }
}
.left-menu{
  width: 200px;
  height: 100vh;
  background: #eee;
}
.main-box{
  display: flex;
  .content-box{
    flex: 1;
  }
}
</style>
