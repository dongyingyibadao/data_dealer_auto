## 🚀 快速上传到 GitHub 并使用

### 📤 第一步：上传代码到 GitHub

**1. 在 GitHub 网站上创建新仓库**
   - 访问：https://github.com/new
   - Repository name: `data_dealer_auto`
   - 选择 Public 或 Private
   - ❌ 不要勾选 "Add a README file"
   - 点击 "Create repository"

**2. 在终端运行以下命令**（替换 `YOUR_USERNAME` 为你的 GitHub 用户名）

```bash
cd /home/dongyingyibadao/data_dealer_auto

# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/data_dealer_auto.git

# 推送代码
git branch -M main
git push -u origin main
```

**3. 输入 GitHub 凭证**
   - Username: 你的 GitHub 用户名
   - Password: Personal Access Token（不是 GitHub 密码）
   
   🔑 如何获取 Token：
   - GitHub → Settings → Developer settings → Personal access tokens
   - Generate new token → 勾选 `repo` → 生成并复制

---

### 📥 拉取代码（在其他电脑或目录）

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/data_dealer_auto.git
cd data_dealer_auto

# 安装环境
conda create -n data_dealer python=3.10
conda activate data_dealer
pip install lerobot Pillow numpy torch openai
```

---

### 🔄 代码更新操作

**我修改了代码，要上传到 GitHub：**

```bash
cd /home/dongyingyibadao/data_dealer_auto

git add .                                    # 添加所有修改
git commit -m "描述你的修改"                  # 提交
git push origin main                         # 推送
```

**GitHub 更新了，我要拉取最新代码：**

```bash
cd /home/dongyingyibadao/data_dealer_auto

git pull origin main                         # 拉取更新
```

---

### ⚡ 完整工作流程

```bash
# 早上开始工作
git pull origin main              # 拉取最新代码

# ... 进行开发和修改 ...

# 提交修改
git add .
git commit -m "今日修改内容"
git push origin main              # 推送到 GitHub
```

---

### 🆘 遇到问题？

1. **推送失败？** → 先拉取：`git pull origin main --rebase`，再推送
2. **身份验证失败？** → 使用 Personal Access Token 而不是密码
3. **有冲突？** → 手动编辑冲突文件，然后 `git add .` → `git commit` → `git push`

详细文档：查看 `GITHUB_GUIDE.md`
