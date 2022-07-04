<template>
  <div class="file-browser-box">
    <el-tree
      ref="treeRef"
      :data="project.fileList"
      :expand-on-click-node="false"
      :filter-node-method="filterNode"
      :load="loadChildren"
      :props="props"
      :empty-text='"没有数据"'
      lazy
      node-key="uuid"
      v-if="project.fileList.length > 0"
      @node-click='nodeClick'
    >
      <template #default="{ node, data }">
        <div>
          <span
            v-if="data.type == FileType.FOLDER"
            class="iconfont icon-folder"
          ></span>
          <span
            v-else-if="getIsVideo(data)"
            class="iconfont icon--Video-File"
          ></span>
          <span v-else class="iconfont icon-file"></span>
          <span>
          <span>{{ data.name }}</span>
        </span>
        </div>
      </template>
    </el-tree>
  </div>
</template>
<style>
.el-tree {
  @apply bg-transparent !important;
}

.file-browser-box .el-tree-node__content {
  @apply hover:bg-white hover:bg-opacity-20  !important;
}

.file-browser-box{
  @apply w-full overflow-x-auto;
}
.file-browser-box .el-tree-node{
  overflow: visible !important;
}
.file-browser-box .el-tree-node__children{
  overflow: visible !important;
}

/*hover:bg-amber-300*/
</style>

<style lang="scss">
.file-browser-box {
  .is-loading {
    & + .iconfont {
      display: none;
    }
  }
  .is-current{
    color: rgb(248 113 113 / var(--tw-text-opacity));
  }
}
</style>

<script lang="ts" setup>
import type Node from 'element-plus/es/components/tree/src/model/node'
import { FileItem, FileType, useProject } from '/@/store/project'
import { defineProps, watch, ref } from 'vue'
import { genFileObj } from '/@/utils/FileUtils'
import { useService } from '/@/composables'
import { VIDEO_SUFFIX_LIST } from '/@/constants/videoConstants'
const { getChildPaths, getSep } = useService('FileService')
const project = useProject()
const filterNode = (value: string, data: FileItem) => {
  if (!value) return true
  return (data.name as string).indexOf(value) !== -1
}
const prop = defineProps({
  filterValue: String
})

const treeRef = ref(null)
watch(() => prop.filterValue, (val) => {
  treeRef.value!.filter(val)
})
async function loadChildren(node: Node, resolve: (data: FileItem[]) => void) {
  try {
    if (node.level === 0) {
      resolve(project.fileList)
      return false
    }
    const data = node.data
    const childrenPathList = await getChildPaths(data.path)
    const childrenList = []
    for (const childrenPath of childrenPathList) {
      const absoluteChildrenPath = data.path + (await getSep()) + childrenPath
      const fileObj = await genFileObj(absoluteChildrenPath)
      childrenList.push(fileObj)
    }
    resolve(
      childrenList.sort((a: FileItem, b: FileItem) => {
        return (a.type as FileType) - (b.type as FileType)
      })
    )
  } catch (err) {
    console.error(err)
    resolve([])
  }
}

const props = {
  isLeaf
}

function isLeaf(data: FileItem) {
  return data.type !== FileType.FOLDER
}
function getIsVideo(data: FileItem) {
  const index = VIDEO_SUFFIX_LIST.findIndex((suffix) => {
    return suffix === data.suffix
  })
  return index >= 0
}
function nodeClick (node: FileItem) {
  if (getIsVideo(node)) {
    const uuid = getVideoUuid(node)
    if (uuid) {
      project.loadVideo(uuid, null)
    } else {
      console.log('没有找到该视频')
    }
  }
}

function getVideoUuid (node: FileItem) {
  const index = project.videoList.findIndex(video => {
    return video.path === node.path
  })
  if (index >= 0) {
    return project.videoList[index].uuid
  }
  return null
}
</script>
