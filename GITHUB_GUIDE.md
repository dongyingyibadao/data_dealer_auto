# 📚 GitHub 使用指南 - data_dealer_auto

## 🚀 第一次上传到 GitHub

### 步骤 1: 在 GitHub 上创建新仓库

1. 打开浏览器，访问 [https://github.com](https://github.com)
2. 登录你的 GitHub 账号
3. 点击右上角的 `+` → `New repository`
4. 填写仓库信息：
   - **Repository name**: `data_dealer_auto`
   - **Description**: `Automated Pick/Place dataset processing tool for LIBERO`
   - **Visibility**: 选择 `Public` 或 `Private`
   - ⚠️ **不要**勾选 "Initialize this repository with a README"
5. 点击 `Create repository`

### 步骤 2: 推送代码到 GitHub

复制 GitHub 给你的仓库地址（例如：`https://github.com/YOUR_USERNAME/data_dealer_auto.git`）

然后在终端运行：

```bash
cd /home/dongyingyibadao/data_dealer_auto

# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/data_dealer_auto.git

# 推送代码到 GitHub
git branch -M main
git push -u origin main
```

如果遇到身份验证问题，你需要：
- 使用 GitHub Personal Access Token (推荐)
- 或使用 SSH key

**生成 Personal Access Token:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → 勾选 `repo` 权限
3. 复制生成的 token
4. 在推送时使用 token 作为密码

---

## 📥 如何将代码 Pull 下来（首次克隆）

### 在新电脑或新目录克隆代码：

```bash
# 方法 1: HTTPS 方式（推荐新手）
git clone https://github.com/YOUR_USERNAME/data_dealer_auto.git
cd data_dealer_auto

# 方法 2: SSH 方式（需要配置 SSH key）
git clone git@github.com:YOUR_USERNAME/data_dealer_auto.git
cd data_dealer_auto
```

### 安装依赖：

```bash
# 创建 conda 环境
conda create -n data_dealer python=3.10
conda activate data_dealer

# 安装依赖
pip install lerobot
pip install Pillow numpy torch openai
```

### 验证安装：

```bash
python auto_cut_dataset.py --help
```

---

## 🔄 如何更新代码（Push & Pull）

### 场景 1: 你修改了代码，想要上传到 GitHub

```bash
cd /home/dongyingyibadao/data_dealer_auto

# 1. 查看修改了哪些文件
git status

# 2. 添加所有修改的文件（或指定特定文件）
git add .                          # 添加所有文件
# 或
git add file1.py file2.py          # 只添加特定文件

# 3. 提交修改（写清楚修改内容）
git commit -m "描述你的修改，例如：添加了快速模式支持"

# 4. 推送到 GitHub
git push origin main

# 如果遇到冲突，先拉取最新代码：
git pull origin main
# 解决冲突后再推送
git push origin main
```

### 场景 2: GitHub 上的代码更新了，你想要拉取最新代码

```bash
cd /home/dongyingyibadao/data_dealer_auto

# 1. 查看当前状态
git status

# 2. 如果有未提交的修改，先保存或提交
git stash                          # 临时保存修改
# 或
git commit -am "保存当前修改"       # 提交修改

# 3. 拉取最新代码
git pull origin main

# 4. 如果之前使用了 stash，恢复修改
git stash pop
```

### 场景 3: 查看代码历史和版本

```bash
# 查看提交历史
git log
git log --oneline                  # 简洁模式

# 查看某个文件的修改历史
git log --follow filename.py

# 查看某次提交的详细内容
git show COMMIT_HASH

# 回退到某个版本（谨慎使用）
git checkout COMMIT_HASH           # 查看历史版本
git checkout main                  # 返回最新版本
```

---

## 🌿 分支管理（进阶）

### 创建新分支进行开发：

```bash
# 创建并切换到新分支
git checkout -b feature/new-feature

# 在新分支上修改代码
# ... 修改文件 ...

# 提交修改
git add .
git commit -m "新功能：描述"

# 推送新分支到 GitHub
git push origin feature/new-feature

# 切换回主分支
git checkout main

# 合并分支（如果测试通过）
git merge feature/new-feature
git push origin main

# 删除已合并的分支
git branch -d feature/new-feature
```

---

## 📋 常用命令速查表

| 命令 | 说明 |
|------|------|
| `git status` | 查看当前状态 |
| `git add .` | 添加所有修改 |
| `git commit -m "message"` | 提交修改 |
| `git push origin main` | 推送到远程 |
| `git pull origin main` | 拉取最新代码 |
| `git log` | 查看提交历史 |
| `git diff` | 查看未提交的修改 |
| `git stash` | 临时保存修改 |
| `git stash pop` | 恢复保存的修改 |
| `git branch` | 查看分支列表 |
| `git checkout -b branch` | 创建并切换分支 |

---

## 🔧 常见问题解决

### Q1: 推送时提示 "Authentication failed"

**解决方案：使用 Personal Access Token**

```bash
# 使用 token 作为密码
# Username: 你的 GitHub 用户名
# Password: 你的 Personal Access Token（不是 GitHub 密码）
```

### Q2: 推送时提示 "rejected" 或 "non-fast-forward"

**解决方案：先拉取再推送**

```bash
git pull origin main --rebase
git push origin main
```

### Q3: 有冲突怎么办？

```bash
# 1. 拉取代码时会显示冲突文件
git pull origin main

# 2. 手动编辑冲突文件，解决冲突标记（<<<<<<, ======, >>>>>>）
# 3. 标记为已解决
git add 冲突文件.py

# 4. 提交解决结果
git commit -m "解决合并冲突"

# 5. 推送
git push origin main
```

### Q4: 误提交了大文件或敏感信息

```bash
# 撤销最后一次提交（但保留修改）
git reset --soft HEAD~1

# 从暂存区移除文件
git reset HEAD large_file.txt

# 添加到 .gitignore
echo "large_file.txt" >> .gitignore

# 重新提交
git add .
git commit -m "修正提交"
git push origin main
```

---

## 🎯 完整工作流程示例

### 日常开发流程：

```bash
# 1. 早上开始工作，先拉取最新代码
cd /home/dongyingyibadao/data_dealer_auto
git pull origin main

# 2. 进行开发
# ... 修改代码 ...

# 3. 测试你的修改
python auto_cut_dataset.py --end-idx 100 --skip-cutting

# 4. 提交修改
git status                                    # 查看修改了什么
git add auto_cut_dataset.py                   # 添加修改的文件
git commit -m "优化：提升处理速度"             # 提交
git push origin main                          # 推送到 GitHub

# 5. 下班前再次推送（确保代码安全）
git add .
git commit -m "今日工作进度"
git push origin main
```

---

## 📞 需要帮助？

- GitHub 官方文档: https://docs.github.com
- Git 教程: https://git-scm.com/book/zh/v2
- 可视化 Git 学习: https://learngitbranching.js.org/?locale=zh_CN

---

## 📝 快速命令参考

```bash
# === 首次上传 ===
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/data_dealer_auto.git
git branch -M main
git push -u origin main

# === 日常更新（推送）===
git add .
git commit -m "描述修改内容"
git push origin main

# === 日常更新（拉取）===
git pull origin main

# === 克隆到新地方 ===
git clone https://github.com/YOUR_USERNAME/data_dealer_auto.git
```
