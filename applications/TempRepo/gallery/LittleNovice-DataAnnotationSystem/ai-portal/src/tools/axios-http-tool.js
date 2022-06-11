import axios from 'axios'
import global from '../global.js'
import router from '../route.js'
import store from '../store'

let http = axios
http.defaults.baseURL = global.API_BASE_URL // 配置axios请求的地址
http.defaults.headers.post['Content-Type'] = 'application/json; charset=utf-8'
http.defaults.crossDomain = true
http.defaults.withCredentials = true //设置cross跨域 并设置访问权限 允许跨域携带cookie信息
http.defaults.headers.common['satoken'] = '' // 设置请求头为 Authorization
// ============================================================================
// 请求拦截器
// ============================================================================
http.interceptors.request.use(
	(config) => {
		// 每次发送请求之前判断vuex中是否存在token
		// 如果存在，则统一在http请求的header都加上token，这样后台根据token判断你的登录情况
		// 即使本地存在token，也有可能token是过期的，所以在响应拦截器中要对返回状态进行判断
		const token = store.state.common.token
		if (token) {
			config.headers.satoken = token
		}
		return config
	},
	(error) => {
		console.error(error)
		return Promise.error(error)
	}
)

// ============================================================================
// 响应拦截器
// ============================================================================
http.interceptors.response.use(
	(response) => {
		// 如果返回的状态码为200，说明接口请求成功，可以正常拿到数据
		// 否则的话抛出错误

		if (response.status === global.CODE_OK) {
			let rsp = response.data
			if (rsp.code === global.CODE_OK) {
				return Promise.resolve(response)
			} else if (rsp.code === global.CODE_OK_FILL_FORM) {
				return Promise.resolve(response)
			} else if (rsp.code === global.CODE_OK_LOGIN) {
				// 跳转到首页
				store.commit('common/setToken', rsp.data.tokenValue)
				store.commit('user/setUserInfo', rsp.data)
				console.log('登录成功')
				router.replace({path: '/home'})
				return Promise.resolve(response)
			} else if (rsp.code === global.CODE_OK_LOGOUT) {
				store.commit('common/resetToken')
				store.commit('user/setUserInfo', {})
				console.log('退出成功')
				router.replace({path: '/login'})
				return Promise.resolve(response)
			} else if (rsp.code == global.CODE_ERROR_NOT_LOGIN) {
				console.log('用户还没有登录')
				console.log(router)
				router.replace({path: '/login'})
				return new Promise(() => {
				})
			} else {
				return Promise.reject(response)
			}
		} else {
			return Promise.reject(response)
		}
	},
	// 服务器状态码不是2开头的的情况
	// 这里可以跟你们的后台开发人员协商好统一的错误状态码
	// 然后根据返回的状态码进行一些操作，例如登录过期提示，错误提示等等
	// 下面列举几个常见的操作，其他需求可自行扩展
	(error) => {
		console.log(error)
		console.log('出现了错误' + JSON.stringify(error))
		return Promise.reject(error)
	}
)

export default http
