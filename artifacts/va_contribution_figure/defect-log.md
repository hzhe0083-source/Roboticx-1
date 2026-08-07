# VA Compound Core Contribution — Visual QA Log

## 目标

一张突出**最重要贡献**的图（论文 Figure 1 候选）：
- 载体（A 区）：constant-memory recursive visual coupling 架构
- **视觉主角（B 区）**：语言 grounding 的架构内因果证据
  （LIBERO Blank +2381% / Swap +607% vs PNPW −0.1% / MetaWorld +0.1%）
- C 区：两级损失简单性（无辅助目标）

## 工具链

- `make_contribution_figure.py`：唯一内容源（坐标精确可控），产出
  `contribution_figure.drawio`（可编辑）+ `preview.html`（embed.diagrams.net 查看）
- `preflight.py`：几何自检（越界 / 真实重叠 / 单行文本溢出）
- 渲染：`rlespinasse/drawio-export` docker（1:1 导出，社区标准）
- QA：三轮像素级视觉审查（子代理，ReadMediaFile 不可用故用 PIL 逐像素验证）

## 审查记录

### 第一轮（Playwright embed 截图 → 弃用）

- embed.diagrams.net 视图缩放不可控（水平 ~1.23×、垂直 ~0.9× 拉伸），
  坐标无法 1:1 → 改用 docker drawio-export 精确渲染。
- P0-1 记忆回环虚线断裂、标签压 ffn 盒
- P0-2 out_a→c_t 连线悬空错位
- P0-3 C_t→sum 线穿过噪声盒
- P1-1 多行文本错乱：根因 `esc()` 把 `&#10;` 双重转义为 `&amp;#10;`
  → 修复：esc 内先保护换行实体
- P1-2 "tokenize" 标签压 qwen 边框 → 盒子改窄留间距

### 第二轮（docker 导出）

- 记忆回环路径已接通；标签仍压竖线（offset −40 不足）→ 改两行标签 + offset −110
- 左上转角缺口 → exit/entry/waypoint 重对齐（exit (0,0.5) → (600,580) → (600,373) → entry (0,0.3)）
- V_t 标签白底压容器左边框（根因：锚点恰在边框 x=620 上）→ 删标签
  （语义由 q_v 盒 "LN(V) → Q_v" 自明），A_t 保留
- 多行文本 3/3/4 行齐全 ✓、tokenize ✓、V/A 分流 ✓、v_θ/integrate ✓

### 第三轮（最终）

- 记忆回环虚线全程连续（含转角），标签完全位于竖线左侧 ✓
- out_a→c_t 路径 + 标签 ✓；C_t→sum 不穿噪声盒 ✓
- 边框 x577-585 在 y186-199 连续（V_t 标签缺口已消除）✓

## 最终状态

- P0 = 0，P1 = 0。全部缺陷三轮内闭环。
- 产物：
  - `contribution_figure.drawio`（30KB，可编辑，drawio / vscode-drawio 直接打开）
  - `contribution_figure.png`（1926×1019，含 30px 白边）
  - `render_export.png`（1866×959，无白边原始导出）
  - `preview.html`（浏览器可编辑预览）

## 已知 P2（接受）

- C_t→sum 入口处 drawio 渲染 8×8 菱形端点伪影
- 记忆回环横线穿越 stack 左边框处边框让位（drawio 标准行为）
- 底部 C 区与 B 区结论横幅之间 ~200px 留白（可后续压缩画布）
