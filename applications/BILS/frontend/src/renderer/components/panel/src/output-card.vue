<template>
  <div class="fixed z-10 label-range-card">
    <card cancelable mode="massage">
      <template #title>
        导出
      </template>
      <template #body>
        <el-tree
          ref="treeRef"
          :data="outputdefine"
          draggable
          node-key="id"
          show-checkbox
          :expand-on-click-node="false"
          default-expand-all
        >
          <template #default="{ node, data }">
            <!--          {{ data.indeterminate || data.check }}-->
        <span
          @dblclick="handleDblClick(data)"
          @keyup.enter.native="data.edit = false"
        >
          <span v-if="!data.edit">{{ data.outName }}</span>
          <div class="w-20">
            <el-input
              ref="edit"
              v-if="data.edit"
              size="small"
              v-model="data.key"
              @blur="data.edit = false"
            />
          </div>
        </span>({{data.name}})
        <span class="flex place-content-center space-x-0.5">
          <a @click="add(node)">
            <span class="iconfont icon-information_add"></span>
          </a><a @click="append(data)">
            <span class="iconfont icon-fangkuai"></span>
          </a>
          <a @click="remove(node, data)">
            <span class="iconfont icon-fangkuai-"></span>
          </a>
        </span>
          </template>
        </el-tree>
      </template>
    </card>
  </div>
</template>

<script setup lang="ts">
import { Card } from '/@/components'
import { nextTick, ref } from 'vue'
import { useOutputDefineStore, OutputDefine } from '/@/store'
const outputDefine = useOutputDefineStore()
const { outputdefine } = outputDefine
const edit = ref()
const handleDblClick = (data) => {
  data.edit = true
  nextTick(() => {
    edit.value.input.focus()
  })
}
const treeRef = ref<InstanceType<typeof ElTree>>()

const newChild = {
  id: (new Date().getTime()).toString(),
  color: '#' + Math.random().toString(16).substr(2, 6).toUpperCase(),
  label: '新建标签',
  edit: true,
  total: 0,
  indeterminate: false,
  check: false,
  children: []
}
const append = (data: OutputDefine | null) => {
  if (!data) treeRef.value.append(newChild)
  else {
    data.children.push(JSON.parse(JSON.stringify(newChild)))
  }
  nextTick(() => {
    edit.value.input.focus()
  })
}
const add = (data: Node) => {
  const parent = data.parent.data
  if (!parent.data) treeRef.value.append(newChild)
  else {
    parent.children.push(JSON.parse(JSON.stringify(newChild)))
  }
  nextTick(() => {
    edit.value.input.focus()
  })
}

const remove = (node: Node, data: OutputDefine) => {
  const parent = node.parent
  const children: OutputDefine[] = parent.data.children || parent.data
  const index = children.findIndex((d) => d.id === data.id)
  children.splice(index, 1)
  if (!treeRef.value.data.length) {
    append()
  }
}

</script>

<style>
</style>
