<template>
  <el-container class="layout-container">
    <el-aside width="220px" class="aside">
      <div class="logo-title">APT 分析平台</div>
      <el-menu
        router
        :default-active="$route.path"
        background-color="#2b2f3a"
        text-color="#bfcbd9"
        active-text-color="#409eff"
        class="custom-menu"
      >
        <el-menu-item index="/analyze">
          <span>🛡️ 智能分析舱</span>
        </el-menu-item>
        <el-menu-item index="/users">
          <span>👥 用户管理</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="header">
        <div class="header-content">
          <span>当前系统状态：<el-tag type="success">运行中</el-tag></span>
          <span>欢迎您，Admin</span>
        </div>
      </el-header>

      <el-main class="main-content">
        <router-view></router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<style>
/* 简单的全局样式，让页面撑满屏幕，显得专业 */
html, body, #app {
  height: 100%;
  margin: 0;
  font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
}

.layout-container {
  height: 100vh;
}

.aside {
  background-color: #2b2f3a;
  color: white;
}

.logo-title {
  height: 60px;
  line-height: 60px;
  text-align: center;
  font-size: 20px;
  font-weight: bold;
  border-bottom: 1px solid #1f2229;
}

.custom-menu {
  border-right: none; /* 去掉菜单自带的丑陋边框 */
}

.header {
  background-color: #ffffff;
  border-bottom: 1px solid #e6e6e6;
  display: flex;
  align-items: center;
}

.header-content {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
  color: #606266;
}

.main-content {
  background-color: #f0f2f5;
  padding: 20px;
}
</style>