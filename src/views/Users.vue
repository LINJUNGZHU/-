<template>
  <div class="users-container">
    <div class="header-action">
      <h3>用户列表</h3>
      <el-button type="primary" @click="dialogVisible = true"> + 新增管理员/用户 </el-button>
    </div>

    <el-table :data="userList" border style="width: 100%" v-loading="loading">
      <el-table-column prop="id" label="ID" width="80" align="center" />
      <el-table-column prop="username" label="用户名" />
      <el-table-column prop="role" label="角色" width="120" align="center">
        <template #default="scope">
          <el-tag :type="scope.row.role === 'admin' ? 'danger' : 'success'">
            {{ scope.row.role.toUpperCase() }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="150" align="center">
        <template #default="scope">
          <el-button type="danger" size="small" @click="handleDelete(scope.row.id)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="dialogVisible" title="新增账号" width="400px">
      <el-form :model="newUser" label-width="80px">
        <el-form-item label="用户名">
          <el-input v-model="newUser.username" placeholder="请输入用户名" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="newUser.password" type="password" placeholder="请输入密码" show-password />
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="newUser.role" placeholder="请选择角色" style="width: 100%">
            <el-option label="管理员 (Admin)" value="admin" />
            <el-option label="普通用户 (User)" value="user" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">取消</el-button>
          <el-button type="primary" @click="createUser">确认添加</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
// 这里是写 JavaScript 逻辑的地方
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'

// --- 变量定义 ---
const userList = ref([]) // 存放表格数据的数组
const loading = ref(false) // 控制表格的加载动画
const dialogVisible = ref(false) // 控制弹窗是否显示
const newUser = ref({ username: '', password: '', role: 'user' }) // 存放新建用户填写的表单数据

// 配置后端的地址
const API_BASE_URL = 'http://127.0.0.1:8000'

// --- 方法：去后端拉取所有用户 ---
const fetchUsers = async () => {
  loading.value = true
  try {
    // 发送 GET 请求给后端的 /users/ 接口
    const response = await axios.get(`${API_BASE_URL}/users/`)
    userList.value = response.data // 把后端返回的数据赋值给表格
  } catch (error) {
    ElMessage.error('获取用户列表失败，请检查后端是否启动')
  } finally {
    loading.value = false
  }
}

// --- 方法：添加新用户 ---
const createUser = async () => {
  if (!newUser.value.username || !newUser.value.password) {
    return ElMessage.warning('用户名和密码不能为空！')
  }

  try {
    // 发送 POST 请求给后端的 /users/ 接口，带上表单数据
    await axios.post(`${API_BASE_URL}/users/`, newUser.value)
    ElMessage.success('用户添加成功！')
    dialogVisible.value = false // 关闭弹窗
    newUser.value = { username: '', password: '', role: 'user' } // 清空表单
    fetchUsers() // 重新拉取一次数据，刷新表格
  } catch (error) {
    // 如果后端抛出 400 错误（比如用户名已存在），这里会捕获并提示
    ElMessage.error(error.response?.data?.detail || '添加失败')
  }
}

// --- 方法：删除用户 ---
const handleDelete = async (id) => {
  // 弹个确认框，防止手抖误删
  ElMessageBox.confirm('确定要删除这个账号吗？', '警告', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning',
  }).then(async () => {
    try {
      // 发送 DELETE 请求给后端
      await axios.delete(`${API_BASE_URL}/users/${id}`)
      ElMessage.success('删除成功')
      fetchUsers() // 刷新表格
    } catch (error) {
      ElMessage.error('删除失败')
    }
  }).catch(() => {
    // 点击取消什么都不做
  })
}

// --- 页面刚加载时，自动执行一次 fetchUsers 拉取数据 ---
onMounted(() => {
  fetchUsers()
})
</script>

<style scoped>
.users-container {
  background: #fff;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.header-action {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
</style>