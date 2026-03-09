import os
import json
import time
import asyncio
import requests
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from neo4j import GraphDatabase # 新增：Neo4j 官方驱动

# ================= 1. 全局配置 (数据库、沙箱、VT与图谱) =================
# 数据库配置
SQLALCHEMY_DATABASE_URL = "sqlite:///./apt_platform.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 沙箱与外部 API 配置
CUCKOO_HOST = "http://192.168.174.132:8090"
POLLING_INTERVAL = 10
VT_API_KEY = "70ddcd2a12741a6ad3f1ff76faab4f2ec7ff6c7bec20892a9fa140722696c33e"

# 🌟 新增：Neo4j 知识图谱配置
# 注意：网页访问是 7474，但代码通过 bolt 协议连接通常是 7687 端口！
NEO4J_URI = "bolt://172.23.216.81:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Gzhuwyy@was"

# 隐藏文件夹
UPLOAD_DIR = ".uploaded_files"
REPORT_DIR = ".sandbox_reports"
VT_REPORT_DIR = ".vt_reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VT_REPORT_DIR, exist_ok=True)


# ================= 2. 数据库模型与数据验证 =================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String, default="user")

Base.metadata.create_all(bind=engine)

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

class UserOut(BaseModel):
    id: int
    username: str
    role: str
    class Config:
        from_attributes = True


# ================= 3. FastAPI 应用与中间件 =================
app = FastAPI(title="APT Analysis Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ================= 4. 深度学习模型架构 (完全对齐训练代码) =================
class CNNConfig:
    MAX_LEN_DYN = 150
    MAX_LEN_STA = 50
    NUM_CHANNELS = 128
    EMBED_DIM = 50
    STA_KERNELS = [2, 3, 4]
    DYN_KERNELS = [3, 4, 5]
    DYN_DILATIONS = [1, 2, 4]
    DROPOUT = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    APT_MAPPING = {
        0: "APT1", 1: "APT10", 2: "APT19", 3: "APT21",
        4: "APT28", 5: "APT29", 6: "APT30", 7: "Dark Hotel",
        8: "Energetic Bear", 9: "Equation Group", 10: "Gorgon Group"
    }

class ImprovedGatedMultimodalFusion(nn.Module):
    def __init__(self, dyn_dim, sta_dim):
        super().__init__()
        self.bn_dyn = nn.BatchNorm1d(dyn_dim)
        self.bn_sta = nn.BatchNorm1d(sta_dim)
        combined_dim = dyn_dim + sta_dim
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2), nn.BatchNorm1d(combined_dim // 2),
            nn.ReLU(), nn.Dropout(CNNConfig.DROPOUT), nn.Linear(combined_dim // 2, 2), nn.Sigmoid()
        )

    def forward(self, f_dyn, f_sta):
        f_dyn_norm, f_sta_norm = self.bn_dyn(f_dyn), self.bn_sta(f_sta)
        combined = torch.cat([f_dyn_norm, f_sta_norm], dim=1)
        weights = self.attention(combined)
        return torch.cat([f_dyn_norm * weights[:, 0].unsqueeze(1), f_sta_norm * weights[:, 1].unsqueeze(1)], dim=1)

class ProposedDualCNN(nn.Module):
    def __init__(self, num_classes, embed_dim, sta_kernels, dyn_kernels, dyn_dilations, dropout_rate):
        super().__init__()
        self.convs_dyn = nn.ModuleList(
            [nn.Conv1d(embed_dim, CNNConfig.NUM_CHANNELS, k, padding=d * (k - 1) // 2, dilation=d) for k, d in
             zip(dyn_kernels, dyn_dilations)])
        self.convs_sta = nn.ModuleList(
            [nn.Conv1d(embed_dim, CNNConfig.NUM_CHANNELS, k, padding=k // 2) for k in sta_kernels])
        self.dropout = nn.Dropout(dropout_rate)
        dyn_out_dim = CNNConfig.NUM_CHANNELS * len(dyn_kernels)
        sta_out_dim = CNNConfig.NUM_CHANNELS * len(sta_kernels)
        self.fusion = ImprovedGatedMultimodalFusion(dyn_out_dim, sta_out_dim)
        self.fc1 = nn.Linear(dyn_out_dim + sta_out_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_classes)

    def forward_branch(self, x, convs):
        x = x.permute(0, 2, 1)
        conved = [F.relu(conv(x)) for conv in convs]
        pooled = [F.adaptive_max_pool1d(c, 1).squeeze(2) for c in conved]
        return torch.cat(pooled, dim=1)

    def forward(self, dyn_emb, sta_emb):
        f_dyn = self.forward_branch(dyn_emb, self.convs_dyn)
        f_sta = self.forward_branch(sta_emb, self.convs_sta)
        fused_features = self.fusion(f_dyn, f_sta)
        x = F.relu(self.bn1(self.fc1(self.dropout(fused_features))))
        x = F.relu(self.fc2(self.dropout(x)))
        return self.out(x)


# ================= 5. 全局加载模型权重与密码本 =================
print("\n[System] 正在初始化双流 CNN 模型环境...")
apt_model = ProposedDualCNN(
    num_classes=len(CNNConfig.APT_MAPPING),
    embed_dim=CNNConfig.EMBED_DIM,
    sta_kernels=CNNConfig.STA_KERNELS,
    dyn_kernels=CNNConfig.DYN_KERNELS,
    dyn_dilations=CNNConfig.DYN_DILATIONS,
    dropout_rate=CNNConfig.DROPOUT
).to(CNNConfig.DEVICE)

MODEL_PATH = "best_dual_cnn_model.pth"
if os.path.exists(MODEL_PATH):
    try:
        apt_model.load_state_dict(torch.load(MODEL_PATH, map_location=CNNConfig.DEVICE))
        apt_model.eval()
        print(f"[System] ✅ 模型权重加载成功！计算设备: {CNNConfig.DEVICE}")
    except Exception as e:
        print(f"[System] ❌ 模型加载失败 (架构不匹配): {e}")

try:
    with open('vocab.pkl', 'rb') as f:
        global_vocab = pickle.load(f)
    global_weights = np.load('weights.npy')
    print("[System] ✅ 真实的词汇表与权重加载成功！")
except Exception as e:
    global_vocab, global_weights = {"<PAD>": 0, "<UNK>": 1}, np.zeros((2, CNNConfig.EMBED_DIM))

def sequence_to_tensor(sequence_str, max_len):
    words = str(sequence_str).split()[:max_len]
    matrix = np.zeros((1, max_len, CNNConfig.EMBED_DIM), dtype=np.float32)
    for j, w in enumerate(words):
        idx = global_vocab.get(w, global_vocab.get("<UNK>", 1))
        matrix[0, j, :] = global_weights[idx]
    return torch.FloatTensor(matrix).to(CNNConfig.DEVICE)


# ================= 6. 特征提取与沙箱相关函数 =================
# ... [为了节省字数，这里省略你已经有的 deduplicate_sequence, compress_api_sequence_fast,
# extract_cuckoo_features, fetch_virustotal_report, extract_static_features, run_local_sandbox 函数]
# 请把你上一版的这几个函数原封不动地保留在这里！...
def deduplicate_sequence(api_list):
    if not api_list: return []
    cleaned_list = [api_list[0]]
    for api in api_list[1:]:
        if api != cleaned_list[-1]: cleaned_list.append(api)
    return cleaned_list

def compress_api_sequence_fast(api_string):
    if not isinstance(api_string, str) or not api_string.strip(): return ""
    apis = api_string.split()
    if not apis: return ""
    compressed_list = [apis[0]]
    for api in apis[1:]:
        if api != compressed_list[-1]: compressed_list.append(api)
    changed = True
    while changed:
        changed = False
        n = len(compressed_list)
        max_pattern_len = min(n // 2, 50)
        i = 0
        while i < len(compressed_list):
            found_match = False
            for p_len in range(1, max_pattern_len + 1):
                if i + 2 * p_len <= len(compressed_list):
                    if compressed_list[i: i + p_len] == compressed_list[i + p_len: i + 2 * p_len]:
                        del compressed_list[i + p_len: i + 2 * p_len]
                        found_match = True
                        changed = True
                        break
            if not found_match: i += 1
    return " ".join(compressed_list)

def extract_cuckoo_features(report_path):
    print("\n[*] 准备提取并压缩动态 API 特征...")
    try:
        with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
            report = json.load(f)
        if not report or 'behavior' not in report: return {"error": "报告中未找到 behavior 字段"}
        processes = report['behavior'].get('processes')
        if not processes or not isinstance(processes, list): return {"error": "无进程数据"}

        raw_api_sequence = []
        for proc in processes:
            calls = proc.get('calls', [])
            if isinstance(calls, list):
                for call in calls:
                    api = call.get('api')
                    if api: raw_api_sequence.append(str(api))

        if not raw_api_sequence: return {"error": "未提取到任何有效的 API 调用"}
        cleaned_sequence = deduplicate_sequence(raw_api_sequence)
        final_sequence = compress_api_sequence_fast(" ".join(cleaned_sequence))
        if not final_sequence.strip(): return {"error": "API 序列压缩后为空"}

        seq_len = len(final_sequence.split())
        print(f"[+] 动态特征提取成功！压缩后 API 序列长度为: {seq_len}")
        return {"status": "success", "api_sequence": final_sequence, "sequence_length": seq_len}
    except Exception as e:
        return {"error": f"解析异常: {str(e)}"}

def fetch_virustotal_report(cuckoo_report_path: str):
    print(f"\n[*] 准备从沙箱报告中提取 Hash 并查询 VirusTotal...")
    try:
        with open(cuckoo_report_path, 'r', encoding='utf-8') as f:
            cuckoo_data = json.load(f)
        file_hash = cuckoo_data.get("target", {}).get("file", {}).get("sha256")
        if not file_hash: return {"error": "Cuckoo报告中无Hash", "hash": "提取失败"}

        print(f"[+] 成功提取到样本 Hash: {file_hash}")
        url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
        headers = {"accept": "application/json", "x-apikey": VT_API_KEY}

        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            vt_data = response.json()
            save_path = os.path.join(VT_REPORT_DIR, f"{file_hash}_vt.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(vt_data, f, ensure_ascii=False, indent=4)
            print(f"[+] VT 静态报告已成功保存至: {save_path}")
            return {"status": "success", "vt_report_path": save_path, "hash": file_hash}
        elif response.status_code == 404:
            return {"error": "VT未收录此样本", "hash": file_hash}
        else:
            return {"error": f"VT API 报错 {response.status_code}", "hash": file_hash}
    except Exception as e:
        return {"error": f"VT解析错误: {str(e)}", "hash": "未知"}

def extract_static_features(vt_report_path):
    print("\n[*] 准备提取静态导入函数(Imported Functions)特征...")
    sequence_words = []
    try:
        with open(vt_report_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        data = json_data.get('data', json_data)
        if not isinstance(data, dict): return {"error": "找不到 data 字段"}

        attributes = data.get('attributes', data)
        pe_info = attributes.get('pe_info', {})
        if isinstance(pe_info, dict):
            import_list = pe_info.get('import_list', [])
            if isinstance(import_list, list):
                for library in import_list:
                    funcs = library.get('imported_functions', [])
                    if isinstance(funcs, list):
                        for func in funcs:
                            if isinstance(func, str) and func.strip():
                                sequence_words.append(func.strip())

        if not sequence_words: return {"error": "未提取到静态导入函数"}
        final_sequence = " ".join(sequence_words)
        seq_len = len(sequence_words)
        print(f"[+] 静态特征提取成功！共提取 {seq_len} 个导入函数。")
        return {"status": "success", "static_sequence": final_sequence, "sequence_length": seq_len}
    except Exception as e:
        return {"error": f"解析异常: {str(e)}"}

async def run_local_sandbox(file_path: str):
    print(f"\n[*] 正在提交文件到 Cuckoo 沙箱: {file_path}")
    filename = os.path.basename(file_path)
    submit_url = f"{CUCKOO_HOST}/tasks/create/file"
    try:
        with open(file_path, "rb") as f:
            r = requests.post(submit_url, files={"file": (filename, f)}, timeout=10)
            r.raise_for_status()
            task_id = r.json().get("task_id")
            if not task_id: return {"error": "无 Task ID"}
    except Exception as e:
        return {"error": str(e)}

    print(f"[*] 成功投递！Task ID: {task_id}，等待沙箱分析...")
    start_time = time.time()
    while True:
        try:
            r = requests.get(f"{CUCKOO_HOST}/tasks/view/{task_id}", timeout=5)
            status = r.json().get("task", {}).get("status", "unknown")
        except: status = "unknown"

        if status == "reported":
            print(f"\n[+] Task {task_id} 分析完成! 耗时: {int(time.time() - start_time)}秒")
            break
        elif status == "failed_analysis":
            return {"error": "Cuckoo analysis failed"}
        if time.time() - start_time > 600: return {"error": "Timeout"}

        print(".", end="", flush=True)
        await asyncio.sleep(POLLING_INTERVAL)

    print("[*] 正在等待沙箱文件系统写入完成 (3秒)...")
    await asyncio.sleep(3)

    report_url = f"{CUCKOO_HOST}/tasks/report/{task_id}/json"
    save_path = os.path.join(REPORT_DIR, f"{filename}_{task_id}.json")
    try:
        r = requests.get(report_url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            print(f"[+] 报告下载成功: {save_path}")
            return {"status": "success", "report_path": save_path, "task_id": task_id}
        else:
            return {"error": f"错误码 {r.status_code}", "task_id": task_id}
    except Exception as e:
        return {"error": f"下载异常: {str(e)}", "task_id": task_id}


# ================= 7. 🌟 核心图谱查询函数 (扩容优化版) 🌟 =================
def query_neo4j_kg(apt_name: str):
    """
    连接 Neo4j 数据库，查询该 APT 组织的恶意软件、工具和攻击模式
    """
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # 💡 修改点 1：把数据库层面的 LIMIT 放大到 100，确保把 TTPs 都能查出来
            cypher_query = """
            MATCH (apt)-[]-(related)
            WHERE apt.name =~ ('(?i).*' + $apt_name + '.*')
            AND (
                any(l IN labels(related) WHERE l CONTAINS 'Malware' OR l CONTAINS '恶意软件') OR
                any(l IN labels(related) WHERE l CONTAINS 'Tool' OR l CONTAINS '工具') OR
                any(l IN labels(related) WHERE l CONTAINS 'Attack_Pattern')
            )
            RETURN labels(related)[0] AS type, related.name AS name
            LIMIT 100
            """
            result = session.run(cypher_query, apt_name=apt_name)

            kg_data = {"malware": [], "tools": [], "attack_patterns": []}

            for record in result:
                lbl = record["type"]
                name = record["name"]
                if not name: continue

                if "Malware" in lbl or "恶意软件" in lbl:
                    kg_data["malware"].append(name)
                elif "Tool" in lbl or "工具" in lbl:
                    kg_data["tools"].append(name)
                elif "Attack" in lbl:
                    kg_data["attack_patterns"].append(name)

            # 💡 修改点 2：在返回前去重，并精准控制每种类型的最大展示数量
            # 这样既能保证 TTP 够多，又不会让前端变成一团乱麻
            return {
                "malware": list(set(kg_data["malware"]))[:15],  # 恶意软件最多 15 个
                "tools": list(set(kg_data["tools"]))[:15],  # 工具最多 15 个
                "attack_patterns": list(set(kg_data["attack_patterns"]))[:25]  # ⚔️ TTP最多放宽到 25 个！
            }

    except Exception as e:
        print(f"\n[!] 🚨 知识图谱连接或查询失败: {e}")
        return None

# ================= 8. 核心分析接口 (终极大融合) =================
@app.post("/api/analyze", tags=["APT分析"])
async def analyze_malware(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())

    # 1. 沙箱与 VT 特征提取流水线
    sandbox_result = await run_local_sandbox(file_location)
    if "error" in sandbox_result:
        return {"status": "error", "filename": file.filename, "hash": "无", "predictions": [], "knowledge_graph_summary": f"❌ 沙箱执行失败: {sandbox_result['error']}"}

    dynamic_features = extract_cuckoo_features(sandbox_result["report_path"])
    if "error" in dynamic_features:
        return {"status": "error", "filename": file.filename, "hash": "无", "predictions": [], "knowledge_graph_summary": f"🛑 动态特征提取失败: {dynamic_features['error']}"}

    vt_result = fetch_virustotal_report(sandbox_result["report_path"])
    final_hash = vt_result.get("hash", "未提取到Hash")
    if "error" in vt_result:
        return {"status": "error", "filename": file.filename, "hash": final_hash, "predictions": [], "knowledge_graph_summary": f"⚠️ VT查询失败: {vt_result['error']}。"}

    static_features = extract_static_features(vt_result["vt_report_path"])
    if "error" in static_features:
        return {"status": "error", "filename": file.filename, "hash": final_hash, "predictions": [], "knowledge_graph_summary": f"🛑 静态特征提取失败: {static_features['error']}。"}

    # 2. 送入深度学习模型
    print("\n[🧠 AI核心] 正在将双模态特征送入 Dual-CNN 进行并行推理...")
    try:
        dyn_tensor = sequence_to_tensor(dynamic_features["api_sequence"], CNNConfig.MAX_LEN_DYN)
        sta_tensor = sequence_to_tensor(static_features["static_sequence"], CNNConfig.MAX_LEN_STA)

        with torch.no_grad():
            outputs = apt_model(dyn_tensor, sta_tensor)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()

        predicted_results = []
        for idx, prob in enumerate(probabilities):
            apt_name = CNNConfig.APT_MAPPING.get(idx, f"未知 APT_{idx}")
            predicted_results.append({
                "apt_group": apt_name,
                "probability": float(prob)
            })

        predicted_results.sort(key=lambda x: x["probability"], reverse=True)
        top_5_predictions = predicted_results[:5]

        top1_apt = top_5_predictions[0]['apt_group']
        top1_prob = top_5_predictions[0]['probability'] * 100
        print(f"[🧠 AI核心] 推理完成！判定结果: {top1_apt} (置信度 {top1_prob:.2f}%)")

        # 3. 🌟 新增：连接图谱查询其背景情报
        print(f"\n[🌐 知识图谱] 正在连接 Neo4j 数据库，深挖 {top1_apt} 组织的底层关联链...")
        kg_data = query_neo4j_kg(top1_apt)

        # 4. 将 AI 预测与图谱情报整合成优美的文案返回给前端
        kg_summary = f"【AI 判定结果】：模型判定该样本属于 {top1_apt} 组织（置信度：{top1_prob:.1f}%）。\n\n"

        if kg_data and (kg_data['malware'] or kg_data['tools'] or kg_data['attack_patterns']):
            kg_summary += f"【威胁图谱溯源】：在 Neo4j 知识库中发现该组织以下历史活动轨迹：\n"
            if kg_data['malware']:
                kg_summary += f"🦠 关联恶意软件：{', '.join(kg_data['malware'][:4])} 等\n"
            if kg_data['tools']:
                kg_summary += f"🛠️ 常见武器工具：{', '.join(kg_data['tools'][:4])} 等\n"
            if kg_data['attack_patterns']:
                kg_summary += f"⚔️ 惯用攻击模式：{', '.join(kg_data['attack_patterns'][:3])} 等"
        elif kg_data is None:
             kg_summary += "（注：图谱数据库连接失败，无法加载延伸情报）"
        else:
             kg_summary += "（注：图谱数据库中暂无该组织具体的 Malware / Tool 关联数据）"
        safe_kg_data = kg_data if kg_data else {"malware": [], "tools": [], "attack_patterns": []}
    except Exception as e:
        print(f"[🧠 AI核心] 推理时发生崩溃: {str(e)}")
        return {"status": "error", "filename": file.filename, "hash": final_hash, "predictions": [], "knowledge_graph_summary": f"❌ 模型推理失败: {str(e)}"}

    return {
        "status": "success",
        "filename": file.filename,
        "hash": final_hash,
        "sandbox_task_id": sandbox_result.get("task_id", "无"),
        "predictions": top_5_predictions,
        "knowledge_graph_summary": kg_summary,
        "kg_data": safe_kg_data  # 👈 【只需要新增这一行】把图谱原始数据传给前端！
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)