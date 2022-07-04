<template>
  <div
    class="inline-flex tool-box space-x-2 items-center justify-center h-10 py-1 pl-4 pr-3 bg-white rounded-lg text-gray-600 z-20"
  >
    <div class="tool-item">
      <span
        class="hover:text-indigo-600 hover:shadow-md iconfont icon-baocun cursor-pointer"
        @click="exportProjectConfig"
      />
      <card
        mode="popup"
        theme="glass"
        class="absolute top-11 left-0 px-1 py-0 tool-tip"
      >
        <template #body>
          <p>保存项目</p>
        </template>
      </card>
    </div>
    <div class="tool-item">
      <span
        class="hover:text-indigo-600 hover:shadow-md iconfont icon-wendangshezhi-peizhiwenjian_tiandi1 before:text-lg cursor-pointer"
        @click="importProjectConfig"
      ></span>
      <card
        mode="popup"
        theme="glass"
        class="absolute top-11 left-0 px-1 py-0 tool-tip"
      >
        <template #body>
          <p>导入项目</p>
        </template>
      </card>
    </div>
    <div class="tool-item">
      <span
        class="hover:text-indigo-600 hover:shadow-md iconfont icon-xiazaijieguowenjian1 before:text-lg cursor-pointer"
        @click="downloadJson"
      />
      <card
        mode="popup"
        theme="glass"
        class="absolute top-11 left-0 px-1 py-0 tool-tip"
      >
        <template #body>
          <p>导出结果</p>
        </template>
      </card>
    </div>
    <div class="w-1 h-5 bg-gray-500 bg-opacity-20" />
    <div class="tool-item flex flex-col space-y-2">
      <span
        class="hover:text-indigo-600 hover:shadow-md iconfont icon-shanchu3 font-bold cursor-pointer"
        @click="deletecurLabel"
      />
      <card mode="popup" theme="glass" class="absolute top-11 left-0 px-1 py-0 tool-tip">
        <template #body>
          <p>删除标签</p>
        </template>
      </card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Card } from '/@/components'
import { useDialog, useService } from '/@/composables'
import { useProject } from '/@/store/project'
import { useLabelsStore } from '/@/store/label'
import { storeToRefs } from 'pinia'
import { exportModel } from '/@/handler/export'
import { ElMessage } from 'element-plus'
import { exportConfigFile, importConfigFile } from '/@/utils/projectUtil'
const projectStore = useProject()
const labelsStore = useLabelsStore()
const { curLabel } = storeToRefs(labelsStore)
const { deletecurLabel } = labelsStore
const { showOpenDialog } = useDialog()
const { getSep, ensureFile } = useService('FileService')

async function downloadJson() {
  const Label = useLabelsStore()
  const labels = Label.getAllLabels()
  const treeArray = labels
  console.log('dowloadJson')
  // console.log(labels)
  if (projectStore.baseConfig.projectPath == null) {
    return
  }
  const jsonDefaultPath =
    projectStore.baseConfig.projectPath + (await getSep()) + 'ai.json'
  await ensureFile(jsonDefaultPath)
  const data = await showOpenDialog({
    title: '选择文件',
    defaultPath: jsonDefaultPath,
    buttonLabel: '确认',
    filters: [{ name: 'json', extensions: ['json'] }],
    properties: ['openFile', 'promptToCreate', 'createDirectory']
  })
  const jsonFilePath = data.filePaths[0]
  if (jsonFilePath == null) {
    return
  }
  await exportModel({
    path: jsonFilePath as string,
    modelType: 'FF',
    outputType: 'json',
    data: treeArray,
    encoding: null
  })
  if (!projectStore.baseConfig.outNoTip) {
    ElMessage({
      message: '导出模型成功',
      type: 'success'
    })
  }
}

async function exportProjectConfig() {
  const jsonDefaultPath =
    projectStore.baseConfig.projectPath +
    (await getSep()) +
    projectStore.baseConfig.projectName +
    '.pConfig'
  await ensureFile(jsonDefaultPath)
  const data = await showOpenDialog({
    title: '选择文件',
    defaultPath: jsonDefaultPath,
    buttonLabel: '确认',
    filters: [{ name: 'pConfig', extensions: ['pConfig'] }],
    properties: ['openFile', 'promptToCreate', 'createDirectory']
  })
  const filePath = data.filePaths[0]
  if (filePath == null) {
    return
  }
  await exportConfigFile(filePath)
  ElMessage({
    message: '导出项目成功',
    type: 'success'
  })
}

async function importProjectConfig() {
  const data = await showOpenDialog({
    title: '选择文件',
    buttonLabel: '导入',
    filters: [{ name: 'pConfig', extensions: ['pConfig'] }],
    properties: ['openFile']
  })
  const filePath = data.filePaths[0]
  if (filePath == null) {
    return
  }
  await importConfigFile(filePath)
}
</script>

<style scoped>
.tool-item .tool-tip {
  border-radius: 5px !important;
  transform: translateX(-50%);
  left: 50%;
  margin-top: 0 !important;
}
.tool-item{
  position: relative;
  white-space: nowrap;
}
.tool-item:hover .tool-tip {
  @apply visible;
}
.tool-item .tool-tip {
  @apply invisible;
}
</style>
