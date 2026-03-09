<template>
  <div class="analyze-container">
    <div class="header">
      <h2>🛡️ APT 恶意软件智能分析舱</h2>
      <p>请上传待分析的样本文件，系统将自动进行沙箱动态分析与静态特征提取。</p>
    </div>

    <el-card class="upload-card" v-if="!analysisResult && !isAnalyzing" shadow="hover">
      <el-upload
        class="upload-demo"
        drag
        action="#"
        :auto-upload="true"
        :show-file-list="false"
        :http-request="uploadFile"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          拖拽文件到此处，或 <em>点击上传</em>
        </div>
      </el-upload>
    </el-card>

    <el-card class="loading-card" v-if="isAnalyzing" shadow="always">
      <div v-loading="isAnalyzing" element-loading-text="正在执行沙箱分析与多模态特征提取，请稍候..." style="height: 200px;">
      </div>
    </el-card>

    <div v-if="analysisResult" class="result-area">
      <el-button type="primary" plain @click="resetAnalysis" class="reset-btn">
        ← 分析另一个样本
      </el-button>

      <el-row :gutter="20">
        <el-col :span="12">
          <el-card shadow="hover" class="result-card">
            <template #header>
              <div class="card-header">
                <span>🎯 双CNN模型预测结果 (Top 5)</span>
              </div>
            </template>
            <div class="prediction-item" v-for="(item, index) in analysisResult.predictions" :key="index">
              <div class="apt-name">{{ item.apt_group }}</div>
              <el-progress
                :percentage="(item.probability * 100).toFixed(1)"
                :color="customColors"
                :stroke-width="15"
              />
            </div>
          </el-card>
        </el-col>

        <el-col :span="12">
          <el-card shadow="hover" class="result-card">
            <template #header>
              <div class="card-header">
                <span>🧠 知识图谱关联分析</span>
              </div>
            </template>
            <el-descriptions :column="1" border>
              <el-descriptions-item label="样本名称">{{ analysisResult.filename }}</el-descriptions-item>
              <el-descriptions-item label="提取 Hash">{{ analysisResult.hash }}</el-descriptions-item>
              <el-descriptions-item label="威胁情报">
                <el-alert :title="analysisResult.knowledge_graph_summary" type="warning" :closable="false" style="white-space: pre-wrap; line-height: 1.5;"/>
              </el-descriptions-item>
            </el-descriptions>
          </el-card>
        </el-col>
      </el-row>

      <el-row style="margin-top: 20px;">
        <el-col :span="24">
          <el-card shadow="hover">
            <template #header>
              <div class="card-header">
                <span>🕸️ 威胁情报图谱可视化 (Neo4j 动态链路)</span>
              </div>
            </template>
            <div ref="graphContainer" style="width: 100%; height: 500px;"></div>
          </el-card>
        </el-col>
      </el-row>

    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import * as echarts from 'echarts' // 🌟 引入 ECharts

// --- 变量定义 ---
const isAnalyzing = ref(false)
const analysisResult = ref(null)
const graphContainer = ref(null) // 绑定绘图 Div
let chartInstance = null // Echarts 实例

const API_BASE_URL = 'http://127.0.0.1:8000'

const customColors = [
  { color: '#f56c6c', percentage: 100 },
  { color: '#e6a23c', percentage: 60 },
  { color: '#5cb87a', percentage: 20 },
]

// --- 方法：自定义文件上传 ---
const uploadFile = async (options) => {
  const file = options.file
  isAnalyzing.value = true

  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await axios.post(`${API_BASE_URL}/api/analyze`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 600000
    })

    ElMessage.success('分析完成！')
    analysisResult.value = response.data

    // 🌟 数据绑定完后，等待 DOM 渲染完毕，开始画知识图谱
    nextTick(() => {
      drawKnowledgeGraph(
        response.data.predictions[0].apt_group, // 取 Top1 为中心点
        response.data.kg_data // 传入后端的图谱结构数据
      )
    })

  } catch (error) {
    ElMessage.error('上传或分析失败，请检查后端运行状态')
    console.error(error)
  } finally {
    isAnalyzing.value = false
  }
}

// --- 🌟 方法：绘制实体关系图 ---
const drawKnowledgeGraph = (centerAptName, kgData) => {
  if (!graphContainer.value) return
  if (chartInstance) chartInstance.dispose() // 如果有旧图则销毁

  chartInstance = echarts.init(graphContainer.value)

  const nodes = []
  const links = []

  // 1. 添加核心节点 (APT组织本身)
  nodes.push({
    id: centerAptName,
    name: centerAptName,
    category: 0,
    symbolSize: 70, // 中心点画大一点
  })

  // 如果没有数据，画个孤立的点就结束
  if (!kgData) return

  // 辅助函数：将后端发来的列表转化为 ECharts 的节点和连线
  const buildNodesAndLinks = (dataList, categoryIndex, relationName) => {
    if (!dataList || dataList.length === 0) return
    dataList.forEach(item => {
      const nodeId = item + '_' + categoryIndex // 拼个后缀防止重名导致报错

      // 添加节点
      nodes.push({
        id: nodeId,
        name: item,
        category: categoryIndex,
        symbolSize: 45 // 周围节点稍微小点
      })

      // 添加连线（从 APT 指向关联实体）
      links.push({
        source: centerAptName,
        target: nodeId,
        value: relationName
      })
    })
  }

  // 2. 灌入三种类别的数据
  buildNodesAndLinks(kgData.malware, 1, '使用恶意软件')
  buildNodesAndLinks(kgData.tools, 2, '使用工具')
  buildNodesAndLinks(kgData.attack_patterns, 3, '常见攻击模式')

  // 3. 配置 ECharts
  const option = {
    tooltip: { trigger: 'item' },
    legend: {
      data: ['APT 组织', 'Malware (恶意软件)', 'Tool (工具)', 'Attack Pattern (战术/模式)']
    },
    color: ['#ee6666', '#91cc75', '#fac858', '#73c0de'], // 定义四种类型颜色
    series: [
      {
        type: 'graph',
        layout: 'force',
        force: {
          repulsion: 600, // 节点之间的斥力大小
          edgeLength: [100, 200] // 连线的长度范围
        },
        roam: true, // 允许鼠标拖拽和缩放图谱
        draggable: true, // 允许拖拽单个节点
        label: {
          show: true,
          position: 'right',
          formatter: '{b}' // 节点上显示节点名字
        },
        categories: [
          { name: 'APT 组织' },
          { name: 'Malware (恶意软件)' },
          { name: 'Tool (工具)' },
          { name: 'Attack Pattern (战术/模式)' }
        ],
        edgeSymbol: ['none', 'arrow'], // 连线末端显示箭头
        edgeLabel: {
          show: true,
          fontSize: 12,
          formatter: "{c}" // 连线上显示关系名
        },
        data: nodes,
        links: links,
      }
    ]
  }

  // 4. 渲染图表！
  chartInstance.setOption(option)
}

const resetAnalysis = () => {
  analysisResult.value = null
}
</script>

<style scoped>
.analyze-container {
  padding: 10px;
}
.header {
  margin-bottom: 30px;
  color: #303133;
}
.header p {
  color: #909399;
  font-size: 14px;
}
.upload-card {
  text-align: center;
  padding: 40px;
}
.loading-card {
  margin-top: 20px;
}
.result-area {
  margin-top: 20px;
}
.reset-btn {
  margin-bottom: 20px;
}
.result-card {
  height: 400px;
}
.prediction-item {
  margin-bottom: 15px;
}
.apt-name {
  font-weight: bold;
  margin-bottom: 5px;
  color: #606266;
}
</style>