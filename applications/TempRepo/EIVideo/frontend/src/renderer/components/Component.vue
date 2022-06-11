<script setup lang="ts">
import { UseDraggable as Draggable } from '@vueuse/components'
</script>

<template>

    <Draggable
      v-slot="{ x, y }"
      p="x-4 y-2"
      border="~ gray-400 rounded"
      shadow="~ hover:lg"
      class="fixed bg-$vt-c-bg select-none cursor-move z-10"
      :prevent-default="true"
      storage-key="vueuse-draggable" storage-type="session"
    >
      Renderless component
      <div class="text-xs opacity-50">
        Position persisted in sessionStorage
      </div>
      <div class="text-sm opacity-50">
        {{ Math.round(x) }}, {{ Math.round(y) }}
      </div>
    </Draggable>
</template>
