import { createMemoryHistory, createRouter } from 'vue-router'
import { Timeline, LineLabel, Control, Component, OutputCard } from '/@/components'
import { FileTest, Label } from '/@/views'

const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    {
      path: '/fileTest',
      component: OutputCard
    },
    {
      path: '/Component',
      component: Component
    },
    {
      path: '/',
      component: Label
    }
  ]
})

export default router
