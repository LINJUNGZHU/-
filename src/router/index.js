import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/analyze' // 默认跳转到分析页面
  },
  {
    path: '/analyze',
    component: () => import('../views/Analyze.vue')
  },
  {
    path: '/users',
    component: () => import('../views/Users.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router