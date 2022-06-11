import {createStore} from 'vuex'

import createPersistedState from 'vuex-persistedstate'

export default createStore({
	modules: {
		user: {
			state: {
				userInfo: {}
			},
			mutations: {
				setUserInfo(state, userInfo) {
					state.userInfo = userInfo
				},
				resetUserInfo(state) {
					state.userInfo = {}
				}
			},
			namespaced: true
		},
		common: {
			state: {
				token: null
			},
			mutations: {
				setToken(state, token) {
					state.token = token
				},
				resetToken(state) {
					state.token = null
				}
			},
			namespaced: true
		}
	},
	plugins: [
		createPersistedState({
			modules: ['user', 'common']
		})
	]
})
