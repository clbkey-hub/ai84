# 命令执行记录 Log

> 记录 AI 助手在终端执行的每条命令。格式：时间 | 目的 | 命令 | 结果

---

## 历史命令补录（本次会话，2026-07-30）

| # | 时间 | 目的 | 命令 | 结果 |
|---|------|------|------|------|
| 1 | 2026-07-30 21:37 | 下载初赛数据集到数据盘（后台） | `cd /root/autodl-tmp && curl -L '<平台downloadFile API>' -H 'Authorization: <token>' -o 初赛数据集.zip --retry 3 -sS -w 'HTTP:%{http_code} SIZE:%{size_download}\n' > download.log 2>&1` | ✅ HTTP 200，3,369,582,475 字节，完整 |
| 2 | 2026-07-30 21:38 | 检查下载进度 | `sleep 20; ls -l /root/autodl-tmp/; cat download.log` | ✅ 确认下载中（1.28GB/3.14GB） |
| 3 | 2026-07-30 | 校验下载完成 | `ls -l 初赛数据集.zip; cat download.log` | ✅ 大小与服务器一致 |
| 4 | 2026-07-30 | 首次解压（UTF-8） | `unzip -O UTF-8 -q 初赛数据集.zip -d 初赛数据集` | ⚠️ 解压成功但中文目录名 GBK 乱码 |
| 5 | 2026-07-30 | 查看解压结果 | `ls -R 初赛数据集; find ... \| wc -l; du -sh` | 138 个文件，3.2G，目录名乱码 |
| 6 | 2026-07-30 | 删除乱码目录并用 GBK 重新解压 | `rm -rf 初赛数据集 && unzip -O GBK -q 初赛数据集.zip -d 初赛数据集` | ✅ 用户确认解压完成，中文目录正常 |
| 7 | 2026-07-30 | 查看数据集结构与训练集样例 | `ls -R \| head -40; head -c 3000 训练集/初赛_训练集.jsonl` | ✅ 训练集 jsonl + 测试集 36 个视频 + 提交说明 |

## 命令记录（从建立 Log 起）

| # | 时间 | 目的 | 命令 | 结果 |
|---|------|------|------|------|
| 8 | 2026-07-30 | 检查GPU/Python/磁盘环境 | `nvidia-smi; python --version; pip list \| grep torch; df -h; conda env list` | ✅ 4090D 24G；Python3.8+torch2.0过旧需新环境 |
| 9 | 2026-07-30 | 统计训练集8字段标签枚举与分布 | `python -c "<统计脚本>" → label_schema.json` | ✅ 实际100条样本；分布极不均衡 |
| 10 | 2026-07-30 | 创建 Python3.10 conda 环境（后台） | `conda create -n vl python=3.10 -y` | ✅ ENV_OK |
| 11 | 2026-07-30 | 激活环境并设置pip清华源 | `conda activate vl && pip config set global.index-url <清华源>` | ✅ READY |
| 12 | 2026-07-30 | 安装依赖（后台） | `pip install torch torchvision transformers accelerate qwen-vl-utils[decord] modelscope pillow` | ✅ PIP_OK |
| 13 | 2026-07-30 | 固定transformers=4.51.3并验证导入 | `pip install transformers==4.51.3 && python -c "from transformers import Qwen2_5_VLForConditionalGeneration"` | ✅ IMPORT_OK |
| 14 | 2026-07-30 | 下载Qwen2.5-VL-7B模型（后台，~16GB） | `modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct` | ✅ MODEL_OK 16G |
| 15 | 2026-07-30 | 训练集前10条推理自测 | `python infer.py --video_dir 训练集/videos --out train_pred_10.jsonl --limit 10` | ✅ 跑通，每条3-8s |
| 16 | 2026-07-30 | 自测评估 | `head -10 训练集.jsonl > gt_10.jsonl && python eval.py --pred train_pred_10.jsonl --gt gt_10.jsonl` | ✅ Avg0.52 预估36.31/70 |
| 17 | 2026-07-30 | 36条测试集正式推理 | `python infer.py --video_dir 测试集/videos --out submit.jsonl` | ✅ 36条完成 |
| 18 | 2026-07-30 | 校验submit.jsonl格式合规 | `python -c "<校验脚本>"` | ✅ 36行/字段全合规 |
| 19 | 2026-08-01 | 检查GPU/内存/磁盘状态 | `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader && free -h && df -h` | ✅ 4090D空闲 24G显存；755G内存可用559G；磁盘50G用22G |
