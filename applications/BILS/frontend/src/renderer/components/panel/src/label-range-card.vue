<template>
  <Draggable
    :initial-value="{ x: 432, y: 485 }"
    v-slot="{ x, y }"
    p="x-4 y-2"
    border="~ gray-400 rounded"
    shadow="~ hover:lg"
    class="fixed bg-$vt-c-bg select-none cursor-move"
    style='z-index: 99999;'
    storage-key="vueuse-draggable" storage-type="session"
    v-show='isShow'
  >
    <div class="fixed z-10 label-range-card">
      <card mode="popup" :customClass="customClass" theme="glass">
      <template #body>
        <item :customClass="itemClass"/>
      </template>
    </card>
  </div>
  </Draggable>
</template>

<script setup lang="ts">
import { Item } from '/@/components/content'
import { Card } from '/@/components'
import { UseDraggable as Draggable } from '@vueuse/components'
import { ref } from 'vue'
import { bus } from '/@/utils/busUtils'
import { CHANGE_MENU_POPUP_SHOW, CHANGE_SEGMENT_SELECT_MENU } from '/@/constants'
const customClass = { container: 'w-80 py-4' }
const itemClass = 'space-y-2'
const isRangeCardShow = ref(false)

const isShow = ref(true)

bus.on(CHANGE_SEGMENT_SELECT_MENU, (flag) => {
  isShow.value = flag
})

</script>

<style>
.label-range-card .text-white{
  @apply text-gray-600;
}
.label-range-card .el-divider {
  @apply border-none hidden !important;
}

.label-range-card .total {
  @apply bg-opacity-25  !important;
}

.label-range-card .button {
  @apply bg-white bg-opacity-20  !important;
}
.hidden-range-card{
  @apply hidden;
}
.submit{
  @apply hover:shadow-none  hover:shadow-lg !important;
}
</style>
