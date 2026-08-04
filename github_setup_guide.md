# GitHub 连接配置指南

## 一、SSH 方式（推荐，本次最终采用）

### 1. 生成密钥对

```bash
ssh-keygen -t ed25519 -C "github-ssh-key" -f ~/.ssh/id_ed25519 -N ""
```

| 参数 | 含义 |
|------|------|
| `-t ed25519` | 密钥类型，ED25519 比 RSA 更快更安全 |
| `-C "github-ssh-key"` | 备注标签，方便在 GitHub 上识别 |
| `-f ~/.ssh/id_ed25519` | 密钥存储路径 |
| `-N ""` | 无密码保护（生产环境建议设密码） |

生成两个文件：
- `~/.ssh/id_ed25519` —— **私钥**（等同于密码，绝不能分享）
- `~/.ssh/id_ed25519.pub` —— **公钥**（贴到 GitHub）

### 2. 查看密钥

```bash
# 查看私钥
cat ~/.ssh/id_ed25519

# 查看公钥
cat ~/.ssh/id_ed25519.pub

# 从私钥重新导出公钥
ssh-keygen -y -f ~/.ssh/id_ed25519
```

### 3. 配置 SSH Config

```bash
cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
EOF
```

### 4. 将公钥添加到 GitHub

1. 复制公钥内容：`cat ~/.ssh/id_ed25519.pub`
2. 打开 https://github.com/settings/ssh/new
3. Title 随便填（如 `autodl-server`）
4. Key 粘贴公钥，点击 Add SSH Key

### 5. 验证连接

```bash
ssh -T git@github.com
# 成功输出：Hi clbkey-hub! You've successfully authenticated...
```

### 6. 克隆/推送

```bash
# 克隆仓库
git clone git@github.com:clbkey-hub/ai84.git

# 推送
git remote add origin git@github.com:clbkey-hub/ai84.git
git push -u origin main
```

---

## 二、HTTPS + Token 方式（备选）

### Classic Token（推荐用这个）

1. 打开 https://github.com/settings/tokens
2. 选择 **Tokens (classic)** → **Generate new token (classic)**
3. Note 填备注（如 `ai84-token`）
4. **必须勾选 `repo`**（以及所有子选项）
5. 点击 Generate，复制 token（`ghp_` 开头）

### 配置远程仓库

```bash
# 格式：https://用户名:token@github.com/用户名/仓库名.git
git remote set-url origin https://clbkey-hub:ghp_xxxxxx@github.com/clbkey-hub/ai84.git
git push -u origin main
```

---

## 三、踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| `Permission denied (publickey)` | 公钥未添加到 GitHub | 去 Settings → SSH Keys 添加 |
| `403` + Fine-grained token | 细粒度 token 不支持 git push | 改用 Classic token |
| `403` + `x-oauth-scopes: ` 为空 | Classic token 没勾选 `repo` | 重新生成并勾选 `repo` |
| `GnuTLS recv error (-110)` | 国内服务器 HTTPS 到 GitHub TLS 被墙 | 改用 SSH 方式 |
| `Empty reply from server` / 超时 | 网络不稳定 | 多试几次或改用 SSH |

---

## 四、关键信息速查

| 项目 | 值 |
|------|-----|
| GitHub 用户名 | `clbkey-hub` |
| 仓库名 | `ai84` |
| 仓库地址 | https://github.com/clbkey-hub/ai84 |
| SSH 密钥路径 | `~/.ssh/id_ed25519` |
| SSH 公钥指纹 | `SHA256:2rgEix3muMJiVtuEviFnpQYaQbbCu5+8dm6V61PcBVw` |

---

## 五、项目上传 GitHub 完整流程（通用）

### 文件筛选原则

> **只上传构建项目所需的最小代码，不上传运行产生的数据**

| 类型 | 是否上传 | 示例 |
|------|---------|------|
| 源代码 | ✅ 上传 | `.py` `.js` `.ts` `.java` `.go` |
| 配置文件 | ✅ 上传 | `.json` `.yaml` `.toml` `requirements.txt` |
| 小数据集 | ✅ 上传 | 几百 KB 的 csv/jsonl |
| README/文档 | ✅ 上传 | `README.md` |
| 模型文件 | ❌ 忽略 | 几百 MB~几十 GB 的 `.safetensors` `.bin` |
| 视频/音频 | ❌ 忽略 | `.mp4` `.wav` `.avi` |
| 压缩包 | ❌ 忽略 | `.zip` `.tar.gz` |
| 密钥/密码 | ❌ 忽略 | `.env` `.token` 含 token 的文件 |
| 编译产物 | ❌ 忽略 | `__pycache__/` `node_modules/` `*.pyc` |
| 日志 | ❌ 忽略 | `*.log` |

### 完整流程（5 步）

#### 第 1 步：创建 `.gitignore`

```bash
# 大文件
models/
*.mp4
*.zip
*.tar.gz
*.safetensors
*.bin

# 临时文件
__pycache__/
*.pyc
*.log

# 敏感信息
.env
*.token

# 系统文件
.DS_Store
node_modules/
.autodl/
```

#### 第 2 步：初始化并暂存

```bash
git init
git add -A
git status    # 检查清单，确认没有不该上传的文件
```

#### 第 3 步：提交

```bash
git config user.name "你的用户名"
git config user.email "你的邮箱"
git commit -m "init: 项目初始化"
```

#### 第 4 步：连接远程仓库

```bash
# SSH（推荐，一次配置永久免密）
git remote add origin git@github.com:用户名/仓库名.git

# HTTPS + Token（备选）
git remote add origin https://用户名:ghp_xxxx@github.com/用户名/仓库名.git
```

#### 第 5 步：推送

```bash
git push -u origin main
```

### SSH 一次性配置（只需做一次，所有项目共用）

```bash
# 1. 生成密钥
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# 2. 查看公钥
cat ~/.ssh/id_ed25519.pub

# 3. 贴到 GitHub → https://github.com/settings/ssh/new

# 4. 配置 SSH config
cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
EOF

# 5. 验证
ssh -T git@github.com
```
