import {createApp} from 'vue'
import App from './App.vue'
import router from './route.js'
import store from './store.js'
import './assets/font/iconfont.css'
import dayjs from 'dayjs'
import './assets/css/index.css'
/*async function init() {
	let DICT = null
	try {
		DICT = await loadDict()
	} finally {
		const vm = createApp(App)
		vm.use(store).use(router).mount('#app')
		vm.config.globalProperties.$dict = DICT
		vm.config.globalProperties.$dayjs = dayjs
	}
}*/

const vm = createApp(App)
vm.use(store).use(router).mount('#app')
vm.config.globalProperties.$dayjs = dayjs
