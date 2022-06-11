<template>
  <div class="main-wrap bg-gray-50">
    <router-view v-slot="{ Component }">
      <keep-alive>
        <component :is="Component" v-if="$route.meta.keepAlive" :key="$route.path"></component>
      </keep-alive>
      <component :is="Component" v-if="!$route.meta.keepAlive"></component>
    </router-view>
  </div>
</template>
<script>
export default {}
</script>

<style lang="less">
.main-wrap {
  min-height: 100vh;
}
</style>
