import global from '@/tools/global.js'
import axios from 'axios'
import token from './token-tool.js'
import {id} from './dom-tool'

axios.defaults.baseURL = global.API_BASE_URL // 配置axios请求的地址
axios.defaults.headers.post['Content-Type'] = 'application/json; charset=utf-8'
axios.defaults.crossDomain = true
axios.defaults.withCredentials = true //设置cross跨域 并设置访问权限 允许跨域携带cookie信息
axios.defaults.headers.common['satoken'] = '' // 设置请求头为 Authorization

// 请求拦截器
axios.interceptors.request.use(
	(config) => {
		// 每次发送请求之前判断vuex中是否存在token
		// 如果存在，则统一在http请求的header都加上token，这样后台根据token判断你的登录情况
		// 即使本地存在token，也有可能token是过期的，所以在响应拦截器中要对返回状态进行判断
		const user = token.get()
		if (user) {
			config.headers.satoken = user.tokenValue
		}
		return config
	},
	(error) => {
		return Promise.error(error)
	}
)

// 响应拦截器
axios.interceptors.response.use(
	(response) => {
		// 如果返回的状态码为200，说明接口请求成功，可以正常拿到数据
		// 否则的话抛出错误

		if (response.status === global.CODE_OK) {
			let rsp = response.data

			if (rsp.code === global.CODE_OK_LOGIN) {
				// 跳转到首页
				token.set(rsp.data)
				setTimeout("javascript:location.href='/portal/index.html'", 0)
			} else if (rsp.code === global.CODE_OK_LOGOUT) {
				token.clear()
				setTimeout("javascript:location.href='/portal/user/user_login.html'", 0)
			} else if (rsp.code == global.CODE_ERROR_NOT_LOGIN) {
				setTimeout("javascript:location.href='/portal/user/user_login.html'", 0)
				return
			} else if (rsp.code == global.CODE_ERROR_FORM_VALID) {
				for (const [key, value] of Object.entries(rsp.data)) {
					console.log(`${key}: ${value}`)
					id(key + '_tips').innerHTML = value
				}
				return Promise.resolve(response)
			} else if (rsp.code == global.CODE_ERROR_DO_OP) {
				console.log(rsp.message)
				id('tips').innerHTML = rsp.message
				return Promise.resolve(response)
			} else {
				return Promise.resolve(response)
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
