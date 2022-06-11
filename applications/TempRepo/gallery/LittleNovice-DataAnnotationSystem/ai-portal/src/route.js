import {createRouter, createWebHistory} from 'vue-router'
import Home from './views/Home'

const routes = [
	{path: '/', redirect: '/home'},
	{
		path: '/home',
		name: 'home',
		meta: {
			keepAlive: true
		},
		component: Home
	}
]

const router = createRouter({
	history: createWebHistory(process.env.NODE_ENV === 'development' ? '/' : '/'),
	routes,
	scrollBehavior(to, from, savedPosition) {
		if (savedPosition && to.meta.isBack) {
			return savedPosition
		} else {
			if (from.meta.keepAlive && to.meta.isBack) {
				from.meta.savedPosition = document.body.scrollTop
			}
			console.log({left: 0, top: from.meta.savedPosition || 0})
			return {left: 0, top: from.meta.savedPosition || 0}
		}
	}
})

router.afterEach((to, from) => {
	const toDepth = to.path.split('/').length
	const fromDepth = from.path.split('/').length
	if (toDepth < fromDepth) {
		to.meta.isBack = true
	} else {
		to.meta.isBack = false
	}
})

export default router
