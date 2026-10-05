# tests/fixtures 测试夹具

由 `generate.py` 生成，供 `tests/` 各用例复用（对应 docs/gui_test_plan.md §4）。

| 文件 | 用途 | 规格 |
|---|---|---|
| `test_input.png` | 静图转换用例 | PIL 新建 64×48 RGB 渐变图 + 白色色块 |
| `test_input.gif` | 动图转换用例 | 3 帧 GIF，每帧 64×48，帧间色块位置不同，duration=100ms |
| `test_input.avi` | 视频转换用例 | cv2.VideoWriter 生成，320×240、30fps、60 帧（fourcc 优先 MJPG，失败回退 mp4v），画面含移动白色色块 |
| `unsupported.txt` | 不支持格式用例 | 一行文本 |

## 用法（幂等：重复运行覆盖生成，不报错）

项目根目录执行一行命令：

```powershell
& ".venv\Scripts\python.exe" tests\fixtures\generate.py --verify
```

`--verify` 生成后逐项校验：文件存在与大小、PNG 尺寸/模式、GIF 帧数与 duration、AVI 可打开/帧数/尺寸/fps，全部通过退出码 0，否则 1。
