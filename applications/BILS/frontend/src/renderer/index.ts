import { createApp } from 'vue'
import App from './App.vue'
import './index.css'
import router from './router'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import vue3videoPlay from 'vue3-video-play'
import './assets/font/iconfont.css'
import './assets/icon/iconfont.css'
import 'vue3-video-play/dist/style.css' // 引入css

const app = createApp(App) // 引入css
app.use(vue3videoPlay)
app.use(ElementPlus)
app.use(router)
app.use(createPinia())
app.mount('#app')
