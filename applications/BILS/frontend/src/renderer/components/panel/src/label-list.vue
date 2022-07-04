<template>
  <!--  {{labels}}-->
  <div class="w-full relative">
    <div class="absolute z-10 left-14 top-1">
      <el-button type="text" v-show="clearable" @click="toggleSelection()">清空</el-button>
    </div>
    <el-table @contextmenu="showContextMenu"
              ref="multipleTableRef"
              :data="labels"
              height="500"
              class="w-full"
              @selection-change="handleSelectionChange"
              @current-change="handleCurrentChange"
              highlight-current-row
              @cell-mouse-enter="enterRow"
    >
      <el-table-column type="selection" v-show="false"/>
      <el-table-column type="index" sortable/>
      <el-table-column label="开始(s)" sortable>
        <template #default="scope">{{ scope.row.range[0] }}</template>
      </el-table-column>
      <el-table-column label="结束(s)" sortable>
        <template #default="scope"> {{ scope.row.range[1] }}</template>
      </el-table-column>
      <el-table-column label="标签">
        <template #default="scope"> {{ scope.row.label.length ? scope.row.label[0].label : '' }}</template>
      </el-table-column>
      <el-table-column label="标签编号">
        <template #default="scope"> {{ scope.row.label.length ? scope.row.label[0].id : '' }}</template>
      </el-table-column>
      <el-table-column type="expand">
        <template #default="scope">
          <p class="font-bold">编号</p>
          <p>{{ scope.row.uuid }}</p>
          <p class="font-bold">标签</p>
          <p v-for="item in scope.row.label" :key="item.id">{{ item.label }}
          </p>
        </template>
      </el-table-column>
    </el-table>
  </div>
  <contextmenu
    ref="contextMenu"
    :offset="{ x: 0, y: 15 }"
    :show="isContextMenuShow"
    @changeContextMenuShow="changeContextMenuShow"
  >
    <div class="context-menu-item" @click="deleteSelectLabel(ROW.uuid)">
      删除
    </div>
  </contextmenu>
</template>
<style>
.el-button--text {
  @apply text-indigo-500 hover:text-indigo-400 !important;
}

.el-table_1_column_2.is-leaf .cell:nth-child(1) {
  @apply hidden !important;
}
</style>
<script lang="ts" setup>
import {ref, computed, watch} from 'vue'
import type {ElTable} from 'element-plus'
import {useService} from '/@/composables'
import {LABELDEFINE, useLabelsStore, useLabelDefineStore, useProject} from '/@/store'
import {storeToRefs} from 'pinia'
import contextmenu from '/@/components/ContextMenu.vue'

const Project = useProject()
const {getSep} = useService('FileService')
const Label = useLabelsStore()
const {deleteLabelById} = Label
const {curLabel, selectedLabel} = storeToRefs(Label)
const clearable = ref(0)

const multipleTableRef = ref<InstanceType<typeof ElTable>>()
const multipleSelection = ref([])
const toggleSelection = (rows?) => {
  if (rows) {
    rows.forEach((row) => {
      multipleTableRef.value!.toggleRowSelection(row, undefined)
    })
  } else {
    multipleTableRef.value!.clearSelection()
  }
}
const handleCurrentChange = (val) => {
  multipleSelection.value = curLabel.value = val
}
const setCurrent = (row) => {
  multipleTableRef.value!.setCurrentRow(row)
}
watch(() => curLabel.value, () => {
  setCurrent(curLabel.value)
})

const handleSelectionChange = (val) => {
  selectedLabel.value = val
  multipleSelection.value = val
  clearable.value = val.length
}
const labels = computed(() => Label.getLabels())
const contextMenu = ref()
const isContextMenuShow = ref(false)
const ROW = ref()

function enterRow(row, column, cell, event) {
  ROW.value = row
}

function showContextMenu(event) {
  contextMenu.value.contextMenuHandler(event)
  isContextMenuShow.value = true
}

function deleteSelectLabel(uuid: string) {
  isContextMenuShow.value = false
  deleteLabelById(uuid)
}

function changeContextMenuShow(flag: boolean) {
  isContextMenuShow.value = flag
}
</script>
