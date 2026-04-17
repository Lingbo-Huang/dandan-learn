# 🚀 部署路线图：从 GitHub Pages 到生产级权限系统

> 当前状态：GitHub Pages 静态托管，无权限控制  
> 目标状态：前端 + Go 后端，完整用户权限系统 + 内容保护

---

## 架构总览

```
用户浏览器
    │
    ▼
Nginx / Cloudflare (反代 + CDN)
    │
    ├──► 前端静态资源 (HTML/CSS/JS)   ← Jekyll 构建产物
    │         │
    │         │ API 请求 (JWT Token)
    │         ▼
    └──► Go 后端服务
              │
              ├── 用户认证 (JWT)
              ├── 权限校验
              ├── 内容访问控制
              └── 水印注入服务
                        │
                        ▼
                   PostgreSQL / SQLite
```

---

## 阶段一：基础部署（1~2天）

### 1.1 服务器选型

| 选项 | 配置 | 费用 | 推荐场景 |
|------|------|------|---------|
| 阿里云 ECS | 2核4G | ~60元/月 | 国内访问快 |
| Hetzner VPS | 2核4G | ~€4/月 | 海外便宜 |
| 腾讯云轻量 | 2核4G | ~50元/月 | 国内备选 |

### 1.2 前端部署

```bash
# 构建 Jekyll 静态站
bundle exec jekyll build
# 产物在 _site/ 目录

# Nginx 配置
server {
    listen 80;
    server_name yourdomain.com;
    root /var/www/dandan-learn/_site;
    index index.html;

    # 所有路由交给前端处理
    location / {
        try_files $uri $uri/ $uri.html =404;
    }

    # API 请求转发到 Go 后端
    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Authorization $http_authorization;
    }
}
```

### 1.3 Go 后端项目结构

```
backend/
├── main.go
├── config/
│   └── config.go          # 配置管理（viper）
├── handler/
│   ├── auth.go            # 登录/注册/邀请码
│   ├── content.go         # 内容访问控制
│   └── watermark.go       # 水印注入
├── middleware/
│   ├── jwt.go             # JWT 校验中间件
│   └── permission.go      # 权限校验中间件
├── model/
│   ├── user.go            # 用户模型
│   └── permission.go      # 权限模型
├── service/
│   ├── auth.go
│   └── watermark.go
└── db/
    └── migrations/        # 数据库迁移文件
```

---

## 阶段二：用户认证系统（3~5天）

### 2.1 数据库设计

```sql
-- 用户表
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    name        VARCHAR(100),
    password    VARCHAR(255),           -- bcrypt hash
    invite_code VARCHAR(50),            -- 使用的邀请码
    role        VARCHAR(20) DEFAULT 'reader',  -- admin / editor / reader
    created_at  TIMESTAMP DEFAULT NOW(),
    last_login  TIMESTAMP
);

-- 邀请码表
CREATE TABLE invite_codes (
    code        VARCHAR(50) PRIMARY KEY,
    created_by  UUID REFERENCES users(id),
    used_by     UUID REFERENCES users(id),
    used_at     TIMESTAMP,
    expires_at  TIMESTAMP,
    max_uses    INT DEFAULT 1,
    use_count   INT DEFAULT 0
);

-- 权限表（细粒度：哪个用户能看哪个 track）
CREATE TABLE permissions (
    user_id     UUID REFERENCES users(id),
    track       VARCHAR(50),  -- ai / llm / agent / infra / quant / wisdom / *
    can_read    BOOLEAN DEFAULT true,
    can_print   BOOLEAN DEFAULT false,
    expires_at  TIMESTAMP,    -- NULL = 永久
    PRIMARY KEY (user_id, track)
);

-- 访问日志（用于水印追溯）
CREATE TABLE access_logs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id),
    path        TEXT,
    ip          INET,
    ua          TEXT,
    accessed_at TIMESTAMP DEFAULT NOW()
);
```

### 2.2 核心 API

```go
// main.go - 路由注册（使用 gin 框架）
r := gin.Default()

// 公开路由
r.POST("/api/auth/login", handler.Login)
r.POST("/api/auth/register", handler.Register)  // 需要邀请码

// 需要认证的路由
auth := r.Group("/api", middleware.JWT())
{
    auth.GET("/auth/me", handler.Me)
    auth.GET("/content/check", handler.CheckAccess)    // 前端校验权限
    auth.GET("/watermark/token", handler.GetWatermarkToken)

    // 管理员路由
    admin := auth.Group("/admin", middleware.RequireRole("admin"))
    {
        admin.POST("/invite", handler.CreateInvite)       // 生成邀请码
        admin.GET("/users", handler.ListUsers)
        admin.PUT("/users/:id/permissions", handler.SetPermissions)
        admin.DELETE("/users/:id", handler.DeleteUser)
    }
}
```

### 2.3 JWT 中间件

```go
// middleware/jwt.go
func JWT() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        // 同时支持 Cookie（网页）和 Header（API）
        if token == "" {
            token, _ = c.Cookie("auth_token")
        }
        if token == "" {
            c.AbortWithStatusJSON(401, gin.H{"error": "未登录"})
            return
        }
        // 验证并解析 JWT
        claims, err := service.ParseJWT(strings.TrimPrefix(token, "Bearer "))
        if err != nil {
            c.AbortWithStatusJSON(401, gin.H{"error": "登录已过期"})
            return
        }
        c.Set("user_id", claims.UserID)
        c.Set("role", claims.Role)
        c.Next()
    }
}
```

---

## 阶段三：权限控制（2~3天）

### 3.1 内容可见性控制

前端在加载文章前先请求权限接口：

```javascript
// 在 post.html 中注入
async function checkAccess() {
    const track = document.body.dataset.track;  // ai/llm/agent...
    const res = await fetch('/api/content/check?track=' + track, {
        headers: { 'Authorization': 'Bearer ' + getToken() }
    });
    if (!res.ok) {
        // 无权限 → 显示遮罩 + 申请访问按钮
        showAccessDenied();
        return;
    }
    // 有权限 → 显示内容
    document.getElementById('a-body').style.display = 'block';
}
```

后端权限校验：

```go
// handler/content.go
func CheckAccess(c *gin.Context) {
    userID := c.GetString("user_id")
    track  := c.Query("track")

    var perm model.Permission
    err := db.Where("user_id = ? AND (track = ? OR track = '*')", userID, track).
        Where("expires_at IS NULL OR expires_at > NOW()").
        First(&perm).Error

    if err != nil || !perm.CanRead {
        c.JSON(403, gin.H{"error": "无访问权限"})
        return
    }
    c.JSON(200, gin.H{"ok": true, "can_print": perm.CanPrint})
}
```

### 3.2 邀请制注册

```
管理员生成邀请码
    │
    ▼
发给被邀请人（链接：https://yourdomain.com/register?code=XXXX）
    │
    ▼
被邀请人填写邮箱 + 密码
    │
    ▼
后端验证邀请码有效性 → 创建账号 → 按邀请码预设分配权限
```

---

## 阶段四：内容保护（2天）

### 4.1 禁止复制的 CSS

```css
/* 普通用户：禁止文字选中和右键 */
.a-body {
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
}

/* 禁止打印（无打印权限时） */
@media print {
    body::before {
        content: "⚠️ 此内容禁止打印";
        display: block;
        font-size: 2em;
        color: red;
        text-align: center;
    }
    .a-body { display: none !important; }
}
```

```javascript
// 禁止右键菜单
document.addEventListener('contextmenu', e => e.preventDefault());
// 禁止常见复制快捷键
document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && ['c','a','s','p','u'].includes(e.key)) {
        e.preventDefault();
    }
});
// 禁止开发者工具检测（初级）
setInterval(() => {
    if (window.outerHeight - window.innerHeight > 200) {
        document.getElementById('a-body').style.display = 'none';
    } else {
        document.getElementById('a-body').style.display = 'block';
    }
}, 1000);
```

> ⚠️ 注意：以上保护只能挡普通用户，无法阻止技术用户（view-source / curl）。真正的保护靠**水印追溯** + **访问控制**。

### 4.2 动态水印（最有效的保护手段）

核心思路：**每个用户看到的内容里都隐藏了他的身份信息，一旦内容泄露可追溯到具体人。**

**方案 A：可见水印（CSS 浮水印）**

```javascript
// 在正文上方叠加半透明水印层
function injectWatermark(username) {
    const canvas = document.createElement('canvas');
    canvas.width = 300; canvas.height = 150;
    const ctx = canvas.getContext('2d');
    ctx.rotate(-25 * Math.PI / 180);
    ctx.font = '13px Inter';
    ctx.fillStyle = 'rgba(0,0,0,0.06)';
    ctx.fillText(username + ' · ' + new Date().toLocaleDateString(), 20, 80);
    const url = canvas.toDataURL();

    const div = document.createElement('div');
    div.style.cssText = `
        position:fixed; inset:0; z-index:500;
        pointer-events:none;
        background-image:url(${url});
        background-repeat:repeat;
    `;
    document.body.appendChild(div);
}
```

**方案 B：不可见水印（零宽字符隐写）**

```go
// service/watermark.go
// 将用户 ID 编码进零宽字符，插入文章正文中
// 截图泄露无法去除，文字复制后隐藏信息仍在

var zwChars = []rune{'\u200b', '\u200c', '\u200d', '\ufeff'}

func InjectInvisibleWatermark(content, userID string) string {
    encoded := encodeToZeroWidth(userID)
    // 每隔 500 字插入一组零宽字符
    runes := []rune(content)
    var result []rune
    for i, r := range runes {
        result = append(result, r)
        if i > 0 && i % 500 == 0 {
            result = append(result, []rune(encoded)...)
        }
    }
    return string(result)
}

func encodeToZeroWidth(s string) string {
    var out strings.Builder
    for _, b := range []byte(s) {
        for i := 0; i < 8; i++ {
            out.WriteRune(zwChars[(b>>uint(i))&1])
        }
    }
    return out.String()
}
```

**方案 C：截图水印（服务端渲染 + Canvas 水印）**

后端将用户信息注入到每页 HTML 响应的 `data-wm` 属性，前端 JS 读取并绘制到 Canvas 覆盖层上，即便截图也带水印。

---

## 阶段五：管理后台（可选，3~5天）

```
管理员界面功能：
├── 用户管理：查看/禁用/删除用户
├── 邀请码管理：生成/吊销/查看使用记录
├── 权限配置：按 track 设置每人的可见性、是否可打印
├── 访问日志：谁在什么时间看了哪些内容
└── 内容统计：各篇文章的阅读量、完读率
```

---

## 技术栈清单

| 层 | 技术选型 |
|----|---------|
| 前端 | Jekyll（现有）+ 少量 Vanilla JS |
| 后端 | Go 1.22 + Gin + GORM |
| 数据库 | PostgreSQL（生产）/ SQLite（本地开发） |
| 认证 | JWT（RS256，非对称密钥） |
| 部署 | Docker Compose（前端 Nginx + 后端 Go + DB） |
| 反代/CDN | Cloudflare（自动 HTTPS + DDoS 保护） |
| 监控 | Prometheus + Grafana（可选） |

---

## Docker Compose 参考

```yaml
# docker-compose.yml
version: '3.9'
services:
  nginx:
    image: nginx:alpine
    volumes:
      - ./_site:/var/www/html:ro
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - backend

  backend:
    build: ./backend
    environment:
      - DB_URL=postgres://postgres:pass@db:5432/dandanlearn
      - JWT_SECRET=${JWT_SECRET}
      - ADMIN_EMAIL=${ADMIN_EMAIL}
    depends_on:
      - db

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: dandanlearn
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## 开发优先级

```
Week 1: Go 后端脚手架 + JWT 认证 + 邀请码注册
Week 2: 权限表 + 内容访问控制 API + 前端对接
Week 3: 水印系统（可见 + 零宽字符）
Week 4: 管理后台 + Docker 部署 + 上线
```

---

*最后更新：2026-04-17 | 维护人：🥚🥚总管*
