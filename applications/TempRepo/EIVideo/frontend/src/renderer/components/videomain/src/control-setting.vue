<template>
  <card mode="popup" class="control-setting-box absolute" theme="glass">
    <template #body>
      <div class="flex flex-col">
        <el-checkbox
          v-model="checkAll"
          :indeterminate="isIndeterminate"
          @change="handleCheckAllChange"
        >全选
        </el-checkbox>
        <el-checkbox-group
          v-model="checkedItem"
        >
          <el-checkbox
            v-for="item in item"
            :key="item"
            :label="item"
            class="w-full"
            @change="handlerCheckedItem(item, $event)"
          >{{ item }}
          </el-checkbox>
        </el-checkbox-group>
      </div>
    </template>
  </card>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { Card } from '/@/components'
import { ElCheckbox, ElCheckboxGroup } from 'element-plus'
import { bus } from '/@/utils/busUtils'
import { CHANGE_AUTO_HIDE_CONTROL, CHANGE_SEGMENT_SELECT_MENU, CHANGE_HIDE_PIC_BOX, CHANGE_DOUBLE_CLICK_INDEX } from '/@/constants/busConstants'

const checkAll = ref(false)
const isIndeterminate = ref(true)
const checkedItem = ref(['缩略图', '双击修改'])
const item = ['自动隐藏', '缩略图', '片选菜单', '双击修改']

const handleCheckAllChange = (val: boolean) => {
  checkedItem.value = val ? item : []
  isIndeterminate.value = false
}
const handleCheckedItemChange = (value: string[]) => {
  const checkedCount = value.length
  checkAll.value = checkedCount === item.length
  isIndeterminate.value = checkedCount > 0 && checkedCount < item.length
}
const handlerCheckedItem = (value: string, flag: boolean) => {
  if (flag) {
    checkedItem.value = checkedItem.value.filter((item) => {
      return item !== value
    })
  } else {
    checkedItem.value.push(value)
  }
  handleCheckedItemChange(checkedItem.value)
}
watch(checkedItem, () => {
  const HideIndex = checkedItem.value.findIndex(item => {
    return item === '自动隐藏'
  })
  bus.emit(CHANGE_AUTO_HIDE_CONTROL, HideIndex >= 0)

  const SegChoseIndex = checkedItem.value.findIndex(item => {
    return item === '片选菜单'
  })
  bus.emit(CHANGE_SEGMENT_SELECT_MENU, SegChoseIndex >= 0)

  const hidePicBoxIndex = checkedItem.value.findIndex(item => {
    return item === '缩略图'
  })
  bus.emit(CHANGE_HIDE_PIC_BOX, hidePicBoxIndex >= 0)

  const doubleClickIndex = checkedItem.value.findIndex(item => {
    return item === '双击修改'
  })
  bus.emit(CHANGE_DOUBLE_CLICK_INDEX, doubleClickIndex >= 0)
}, { deep: true, immediate: true })
</script>

<style lang="scss" scoped>
.control-setting-box {
  width: 200px;
  bottom: 55px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 999;
  display: none;
  opacity: 0;
  transition: all .1s;
  .el-checkbox-group{
    display: flex;
    flex-direction: column;
  }
}
</style>
