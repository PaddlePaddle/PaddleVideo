<template>
  <!--  <div class="bg-amber-100">-->
  <el-form
    ref="ruleFormRef"
    :model="form"
    :rules="rules"
    label-width="120px"
    label-position="top"
  >
    <el-divider content-position="left">项目设置</el-divider>
    <el-form-item label="项目名称" prop="projectName">
      <el-input v-model="form.projectName"></el-input>
    </el-form-item>
    <el-form-item label="项目目录/URL" prop="projectPath">
      <el-input v-model="form.projectPath">
        <template #suffix>
          <el-icon style="height: inherit; cursor: pointer">
            <folder-opened @click="openProjectPathFolder"></folder-opened>
          </el-icon>
        </template>
      </el-input>
    </el-form-item>
    <el-form-item label="数据集目录/URL" prop="dataPath">
      <el-input v-model="form.dataPath">
        <template #suffix>
          <el-icon style="height: inherit; cursor: pointer">
            <folder-opened @click="openDataPathFolder"></folder-opened>
          </el-icon>
        </template>
      </el-input>
    </el-form-item>
    <el-divider content-position="left">导出设置</el-divider>
    <!--    <el-form-item label="采样率">-->
    <!--      <el-input v-model="form.fps"></el-input>-->
    <!--    </el-form-item>-->
    <el-form-item label="导出内容">
      <el-select
        v-model="form.outContent"
        placeholder="please select your zone"
      >
        <el-option label="所有看过部分" value="viewed"></el-option>
        <el-option label="所有标记部分" value="labeled"></el-option>
        <el-option label="全部" value="all"></el-option>
      </el-select>
    </el-form-item>
    <!--    <el-form-item label="导出格式">-->
    <!--      <el-select v-model="form.format" placeholder="Select">-->
    <!--        <el-option-group-->
    <!--          v-for="group in options"-->
    <!--          :key="group.label"-->
    <!--          :label="group.label"-->
    <!--        >-->
    <!--          <el-option-->
    <!--            v-for="item in group.options"-->
    <!--            :key="item.value"-->
    <!--            :label="item.label"-->
    <!--            :value="item.value"-->
    <!--          >-->
    <!--          </el-option>-->
    <!--        </el-option-group>-->
    <!--      </el-select>-->
    <!--    </el-form-item>-->
    <el-form-item label="项目描述">
      <el-input v-model="form.desc" type="textarea"></el-input>
    </el-form-item>
    <el-form-item label="导出格式">
      <el-button type="text" @click="showExportConfig">点击设置</el-button>
    </el-form-item>
    <el-form-item>
      <div class="flex space-x-10">
        <div
          class="hover:shadow-indigo-500/50 hover:text-white hover:shadow-md hover:bg-indigo-600 bg-indigo-600 bg-opacity-90 shadow  inline-flex relative items-center cursor-pointer justify-center h-9 px-5 py-2  rounded-lg"
          @click="onSaveConfig(ruleFormRef)"
        >
          <p class="text-xs font-semibold text-center text-white">保存配置</p>
        </div>
        <div >
          <div
            class="hover:text-indigo-700 hover:bg-indigo-50 hover:shadow-md inline-flex  items-center cursor-pointer justify-center h-9 px-5 py-2 shadow rounded-lg"
            @click="onSubmit(ruleFormRef)"
          >
            <el-icon
              v-show="isSaveLoad"
              class="is-loading"
              style="margin-right: 5px; color: #4e49dc"
            >
              <Loading/>
            </el-icon>
            <p class="text-xs font-semibold text-center text-indigo-500">更新文件</p>
          </div>
        </div>
      </div>
    </el-form-item>
  </el-form>
  <el-dialog title="配置导出格式" v-model="isShowExportConfig">
    <export-config ref="exportConfigRef"></export-config>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="isShowExportConfig = false">取消</el-button>
        <el-button @click="resetExportConfig">重置</el-button>
        <el-button type="primary" @click="confirmExportConfig">确认</el-button>
      </span>
    </template>
  </el-dialog>
  <!--  </div>-->
</template>

<script lang="ts" setup>
import { FolderOpened, Loading } from '@element-plus/icons-vue'
import { useDialog, useService } from '/@/composables'
import getFormConfig from './formConfig'
import type { ElForm } from 'element-plus'
import { nextTick, ref, watch } from 'vue'
import { useProject } from '/@/store/project'
import { loadAllVideo } from '/@/utils/videoUtils'
import { genFileObj } from '/@/utils/FileUtils'
import ExportConfig from '/@/components/content/src/exportConfig.vue'
import { useLabelsStore } from '/@/store'
import { currentExportJson } from '/@/utils/projectUtil'
import { ElMessage } from 'element-plus/es'

const isShowExportConfig = ref(false)
const { form, rules } = getFormConfig()
const isSaveLoad = ref(false)
const projectStore = useProject()

const { getSep } = useService('FileService')
const labelStore = useLabelsStore()
type FormInstance = InstanceType<typeof ElForm>
const ruleFormRef = ref<FormInstance>()
watch(
  projectStore.project,
  function () {
    for (const key in projectStore.baseConfig) {
      // @ts-ignore
      form[key] = projectStore.baseConfig[key]
    }
  },
  { deep: true }
)

const onSubmit = (formEl: FormInstance | undefined) => {
  formEl?.validate(async (valid) => {
    if (!valid) {
      return
    }
    isSaveLoad.value = true
    try {
      projectStore.resetFileList()
      projectStore.resetVideoMeta()
      labelStore.resetProjectLabels()
      projectStore.changeProjectBaseConfig(form)
      await loadAllVideo(form)
      const fileObj = await genFileObj(form.dataPath)
      projectStore.setFileItem(fileObj)
      ElMessage({
        message: '更新文件成功',
        type: 'success'
      })
    } finally {
      isSaveLoad.value = false
    }
    console.log('submit!')
  })
}
const onSaveConfig = (formEl: FormInstance | undefined) => {
  formEl?.validate(async (valid) => {
    if (!valid) {
      return
    }
    projectStore.changeProjectBaseConfig(form)
    ElMessage({
      message: '保存配置成功',
      type: 'success'
    })
    console.log('submit!')
  })
}

const exportConfigRef = ref()
const { showOpenDialog } = useDialog()
const { getGeneralPath } = useService('FileService')
const openFolder = async () => {
  const data = await showOpenDialog({
    title: '选择文件',
    properties: ['promptToCreate', 'openDirectory', 'createDirectory']
  })
  const filePath = await getGeneralPath(data.filePaths[0])
  return filePath
}
const openProjectPathFolder = async () => {
  const filePath = await openFolder()
  form.projectPath = filePath
}
const openDataPathFolder = async () => {
  const filePath = await openFolder()
  form.dataPath = filePath
}

const resetExportConfig = () => {
  console.log(exportConfigRef.value)
  exportConfigRef.value.resetConfig()
}
const confirmExportConfig = async () => {
  const flag = exportConfigRef.value.confirm()
  if (flag) {
    isShowExportConfig.value = false
  }
  if (projectStore.baseConfig.autoSave) {
    const jsonDefaultPath =
      projectStore.baseConfig.projectPath +
      (await getSep()) +
      projectStore.baseConfig.projectName +
      '.pConfig'
    await currentExportJson(jsonDefaultPath)
  }
}
const showExportConfig = () => {
  isShowExportConfig.value = true
  nextTick(() => {
    exportConfigRef.value.show()
  })
}
const options = [
  {
    label: '内置模板',
    options: [
      {
        value: 'json',
        label: 'json'
      },
      {
        value: 'xml',
        label: 'xml'
      },
      {
        value: 'yaml',
        label: 'yaml'
      }
    ]
  },
  {
    label: '自定义模板',
    options: [
      {
        value: 'lanqiubiaozhu',
        label: '篮球标注'
      }
    ]
  }
]
</script>
