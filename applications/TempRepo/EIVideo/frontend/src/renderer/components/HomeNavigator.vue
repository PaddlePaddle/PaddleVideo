<template>
  <div class="sidebar">
    <router-link class="clickable" replace to="/">Test</router-link>
    <router-link class="clickable" replace to="/fileTest">fileTest</router-link>
    <router-link class="clickable" replace to="/Component">Component</router-link>
  </div>
</template>

<style scoped>
.sidebar {
  display: flex;
  background: #9E9E9E;
}

.sidebar a {
  flex-grow: 1;
  color: white;
  text-decoration: none;
  padding: 10px;
  transition-duration: 0.25s;
}

.sidebar a:hover {
  background: rgba(0, 0, 0, 0.25);
}

.sidebar a:last-child {
  border-right: none;
}
</style>
