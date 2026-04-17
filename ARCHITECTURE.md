# 🏗️ 系统架构设计文档

> dandan-learn 权限控制系统  
> 版本：v1.0 | 日期：2026-04-17

---

## 一、核心问题的回答

### 1. 要几个仓库？

**推荐：两个仓库**

```
dandan-learn/          ← 现有仓库（内容仓库）
  ├── ai/week01/...    Markdown 内容文件
  ├── _layouts/        Jekyll 模板
  └── ARCHITECTURE.md  本文档

dandan-learn-server/   ← 新建仓库（Go 后端）
  ├── main.go
  ├── handler/
  ├── middleware/
  └── ...
```

**理由：**
- 内容（MD 文件）和服务端逻辑是两个完全不同的关注点，分开维护更清晰
- 内容仓库可以继续用 GitHub Actions 自动构建静态站
- 后端仓库可以独立部署、独立 CI/CD，不互相干扰
- 如果以后开放部分内容，只需改内容仓库的权限设置即可

---

### 2. MD 内容要存数据库吗？

**不存。MD 文件保持在 Git 仓库，数据库只存「谁能看什么」。**

```
内容的真相来源 → GitHub 仓库（Markdown 文件）
权限的真相来源 → PostgreSQL（用户/权限/日志）
```

**理由：**
- MD 文件天然适合 Git 管理（版本历史、diff、协作）
- 存数据库反而增加复杂度：需要同步机制、全文搜索引擎等
- 后端的职责是「决定用户能不能看」，不是「提供内容」
- 内容由 Jekyll 构建成 HTML，Nginx 直接托管静态文件，性能最好

**内容访问流程：**
```
用户请求文章 → Nginx 检查是否有有效 Cookie
  ├── 无 Cookie → 302 重定向到登录页
  └── 有 Cookie → Go 后端验证 JWT + 权限
        ├── 无权限 → 返回 403 页面
        └── 有权限 → Nginx 放行，返回静态 HTML + 注入水印
```

---

## 二、完整系统架构

```
┌─────────────────────────────────────────────────────────┐
│                        用户浏览器                         │
│  ① 请求页面  ③ 展示内容+水印  ⑤ API 调用（查进度等）    │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTPS
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Cloudflare（CDN + TLS）                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      Nginx（反代）                        │
│                                                         │
│  location /api/          location /auth/                │
│      │                       │                         │
│      ▼                       ▼                         │
│  proxy_pass Go:8080      proxy_pass Go:8080             │
│                                                         │
│  location / （其他所有路径）                              │
│      │                                                  │
│      ├── auth_request /api/auth/check  ← 每次请求先过权限 │
│      │       ├── 401 → redirect /login                  │
│      │       └── 200 → 继续                             │
│      │                                                  │
│      └── root /var/www/dandan-learn/_site  ← 静态文件    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     Go 后端（:8080）                      │
│                                                         │
│  /api/auth/check     权限校验（Nginx auth_request 用）    │
│  /api/auth/login     登录，返回 JWT Cookie               │
│  /api/auth/logout    清除 Cookie                        │
│  /api/auth/register  注册（需邀请码）                    │
│  /api/auth/me        当前用户信息                        │
│                                                         │
│  /api/watermark/js   返回含用户信息的水印 JS 片段          │
│                                                         │
│  /api/admin/*        管理员接口（用户/权限/邀请码）         │
│                                                         │
│  核心模块：                                               │
│  ├── JWT 签发与校验（RS256）                              │
│  ├── 权限矩阵查询（用户×Track）                           │
│  ├── 水印 Token 生成                                     │
│  └── 访问日志记录                                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   PostgreSQL                             │
│                                                         │
│  users          用户账号                                  │
│  invite_codes   邀请码                                   │
│  permissions    用户×Track 权限表                         │
│  sessions       登录会话（可选，用于强制下线）              │
│  access_logs    访问日志（水印追溯）                       │
└─────────────────────────────────────────────────────────┘
```

---

## 三、关键设计决策

### 3.1 权限校验放 Nginx，不放前端 JS

❌ **错误做法**（很多人这么做）：
```javascript
// 前端 JS 校验权限，无权限就隐藏内容
if (!hasPermission) {
  document.getElementById('content').style.display = 'none';
}
```
问题：F12 → Console 两行代码就绕过了。

✅ **正确做法**：利用 Nginx 的 `auth_request` 模块：
```nginx
location / {
    # 每次请求先问 Go 后端「这个用户有没有权限」
    auth_request /api/auth/check;
    # 无权限时 Nginx 直接返回 403，内容根本不下发
    error_page 401 = @login_redirect;
    error_page 403 = @forbidden;
    
    # 权限通过，正常返回静态文件
    root /var/www/dandan-learn/_site;
    try_files $uri $uri/ $uri.html =404;
}

location /api/auth/check {
    internal;  # 只允许 Nginx 内部调用，外部无法直接访问
    proxy_pass http://127.0.0.1:8080;
    proxy_pass_request_body off;
    proxy_set_header Content-Length "";
    proxy_set_header Cookie $http_cookie;
    # Go 后端从 Cookie 拿 JWT → 查权限 → 返回 200/401/403
}
```

这样内容文件在服务端直接拦截，**JS 无法绕过**。

### 3.2 JWT 存 Cookie，不存 localStorage

```
localStorage: JS 可读 → XSS 攻击可偷走 Token
Cookie(HttpOnly + Secure + SameSite=Strict): JS 不可读 → 安全
```

Go 登录接口设置 Cookie：
```go
c.SetCookie(
    "auth_token",        // name
    tokenString,         // value（JWT）
    86400 * 7,           // maxAge（7天）
    "/",                 // path
    "yourdomain.com",    // domain
    true,                // secure（只 HTTPS）
    true,                // httpOnly（JS 不可读）
)
```

### 3.3 Track 粒度权限，不是文章粒度

权限粒度设计：

```
Track 级别（推荐，简单够用）：
  用户 A → 可看：ai, llm
  用户 B → 可看：ai, llm, agent, wisdom
  用户 C → 可看：*（全部）

文章级别（过于复杂，暂不做）：
  用户 A → 可看：ai/week01/*, ai/week02/day01
  ← 管理成本太高，除非有特殊需求
```

Track 对应 URL 路径：
```
/ai/*      → track = "ai"
/llm/*     → track = "llm"
/agent/*   → track = "agent"
/infra/*   → track = "infra"
/quant/*   → track = "quant"
/wisdom/*  → track = "wisdom"
```

### 3.4 水印方案

**两层水印叠加：**

**层 1：可见 Canvas 水印**（防截图传播）
```
Go 后端 /api/watermark/js 接口 → 返回一段 JS
JS 在页面上绘制半透明浮层，内容为「用户名 · 日期 · ID后4位」
截图会带上水印，可追溯到具体人
```

**层 2：不可见零宽字符水印**（防文字复制传播）
```
文章 HTML 由 Nginx sub_filter 模块注入
每隔一段正文插入用户 ID 的零宽字符编码
复制文字后粘贴，肉眼不可见，但可解码出用户 ID
```

两层结合：截图泄露 → 追到人；文字泄露 → 也追到人。

---

## 四、数据库设计

```sql
-- 用户
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    passwd_hash TEXT NOT NULL,              -- bcrypt
    role        TEXT NOT NULL DEFAULT 'reader', -- reader / admin
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_login  TIMESTAMPTZ
);

-- 邀请码
CREATE TABLE invite_codes (
    code        TEXT PRIMARY KEY,
    created_by  UUID REFERENCES users(id),
    note        TEXT,                       -- 备注（给谁用的）
    max_uses    INT NOT NULL DEFAULT 1,
    use_count   INT NOT NULL DEFAULT 0,
    expires_at  TIMESTAMPTZ,               -- NULL = 永不过期
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 邀请记录
CREATE TABLE invite_uses (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code        TEXT REFERENCES invite_codes(code),
    used_by     UUID REFERENCES users(id),
    used_at     TIMESTAMPTZ DEFAULT NOW()
);

-- 权限（用户 × Track）
CREATE TABLE permissions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    track       TEXT NOT NULL,             -- ai/llm/agent/infra/quant/wisdom/*
    can_read    BOOLEAN NOT NULL DEFAULT TRUE,
    expires_at  TIMESTAMPTZ,              -- NULL = 永久
    granted_by  UUID REFERENCES users(id),
    granted_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, track)
);

-- 访问日志
CREATE TABLE access_logs (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID REFERENCES users(id),
    path        TEXT NOT NULL,
    track       TEXT,
    ip          TEXT,
    user_agent  TEXT,
    accessed_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_access_logs_user ON access_logs(user_id, accessed_at DESC);
CREATE INDEX idx_access_logs_time ON access_logs(accessed_at DESC);
```

---

## 五、Go 后端项目结构

```
dandan-learn-server/
├── cmd/
│   └── server/
│       └── main.go              # 启动入口
├── internal/
│   ├── config/
│   │   └── config.go            # 配置（viper 读 .env）
│   ├── db/
│   │   ├── db.go                # 连接池初始化
│   │   └── migrations/          # SQL 迁移文件
│   │       ├── 001_init.sql
│   │       └── 002_add_logs.sql
│   ├── model/
│   │   ├── user.go
│   │   ├── invite.go
│   │   └── permission.go
│   ├── handler/
│   │   ├── auth.go              # login/logout/register/me/check
│   │   ├── watermark.go         # 水印 JS 生成
│   │   └── admin.go             # 管理接口
│   ├── middleware/
│   │   ├── jwt.go               # JWT 解析
│   │   └── require_role.go      # 角色校验
│   └── service/
│       ├── auth.go              # 业务逻辑
│       ├── permission.go        # 权限查询（带缓存）
│       └── watermark.go         # 水印生成
├── pkg/
│   └── jwt/
│       └── jwt.go               # JWT 工具（RS256）
├── web/                         # 极简管理界面（可选）
│   └── admin/
│       └── index.html
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── go.mod
```

---

## 六、核心接口定义

```
POST /api/auth/login
  Body: { email, password }
  Response: 设置 HttpOnly Cookie，返回 { name, role }

POST /api/auth/logout
  Response: 清除 Cookie

POST /api/auth/register
  Body: { email, name, password, invite_code }
  Response: 201 Created

GET  /api/auth/me
  Response: { id, email, name, role, tracks[] }

GET  /api/auth/check
  （Nginx internal 调用，不对外暴露）
  从 Cookie 读 JWT → 查权限 → 200/401/403
  Response Header: X-User-Id, X-User-Name（给水印用）

GET  /api/watermark/js
  Response: 一段 JS，在浏览器绘制水印

POST /api/admin/invites
  Body: { note, max_uses, expires_at, tracks[] }
  Response: { code }

GET  /api/admin/users
GET  /api/admin/users/:id
PUT  /api/admin/users/:id/permissions
DELETE /api/admin/users/:id
GET  /api/admin/logs?user_id=&from=&to=
```

---

## 七、部署方案

### 7.1 目录结构（服务器上）

```
/opt/dandan/
├── frontend/          # Jekyll 构建产物（_site/）
│   └── _site/
├── server/            # Go 二进制
│   └── dandan-server
├── config/
│   ├── nginx.conf
│   └── .env
└── docker-compose.yml
```

### 7.2 Nginx 完整配置

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate     /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    # 静态资源（公开，无需登录）
    location /assets/ {
        root /opt/dandan/frontend/_site;
        expires 7d;
    }

    # 登录页（公开）
    location = /login {
        root /opt/dandan/frontend/_site;
        try_files /login.html =404;
    }

    # API（直接转发，不走 auth_request）
    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Cookie $http_cookie;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # 所有内容页（需要权限）
    location / {
        auth_request /api/auth/check;
        error_page 401 = @do_login;
        error_page 403 = @do_forbidden;

        # 将用户信息传给前端（注入水印用）
        auth_request_set $user_name $upstream_http_x_user_name;
        add_header X-User-Name $user_name;

        root /opt/dandan/frontend/_site;
        try_files $uri $uri/ $uri.html =404;
    }

    location @do_login {
        return 302 /login?next=$request_uri;
    }
    location @do_forbidden {
        return 403 /403.html;
    }

    # Nginx internal 权限检查（不对外暴露）
    location = /api/auth/check {
        internal;
        proxy_pass http://127.0.0.1:8080;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header Cookie $http_cookie;
        proxy_set_header X-Original-URI $request_uri;
    }
}
```

### 7.3 Docker Compose

```yaml
version: '3.9'
services:
  server:
    build: .
    restart: unless-stopped
    ports:
      - "127.0.0.1:8080:8080"    # 只暴露给 Nginx，不对外
    env_file: .env
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: dandanlearn
      POSTGRES_USER: dandan
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dandan"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  pgdata:
```

---

## 八、开发顺序（建议）

```
Week 1：脚手架 + 认证
  Day 1-2: 初始化项目，数据库迁移，用户模型
  Day 3-4: 登录/注册/JWT/Cookie
  Day 5:   /api/auth/check 接口，本地 Nginx 测试

Week 2：权限 + 邀请
  Day 1-2: 权限表，权限查询（带内存缓存）
  Day 3:   邀请码生成与校验
  Day 4-5: 管理接口（用户列表/权限设置）

Week 3：水印
  Day 1-2: Canvas 可见水印 JS 生成
  Day 3-4: 零宽字符隐写注入（Nginx sub_filter 或 Go 中间件）
  Day 5:   水印解码工具（追溯时用）

Week 4：部署 + 收尾
  Day 1-2: Dockerfile + Docker Compose + 服务器配置
  Day 3:   域名 + HTTPS + Cloudflare
  Day 4:   压测 + 安全检查
  Day 5:   上线
```

---

## 九、FAQ

**Q：用户忘记密码怎么办？**  
A：第一版直接管理员重置（admin 接口），后续加邮件重置。

**Q：JWT 过期了怎么办？**  
A：用 Refresh Token 机制，access token 2小时，refresh token 7天存 Cookie，自动续期。

**Q：高并发时权限查询会不会成为瓶颈？**  
A：权限数据变化少，在 Go 内存中用 `sync.Map` 缓存，TTL 5分钟，绰绰有余。

**Q：用户被封号/吊销权限，已有 JWT 还有效怎么办？**  
A：在 DB 维护一个「被吊销的用户 ID 集合」（或 session 表），`/api/auth/check` 每次都查，命中则 401。

**Q：内容更新（新 MD 文件）怎么触发部署？**  
A：GitHub Actions：内容仓库 push → 自动 Jekyll build → rsync 到服务器 `/opt/dandan/frontend/_site/`，Go 后端无感知。

---

*本文档是开发前的架构设计，实现过程中可能调整细节，以代码为准。*
