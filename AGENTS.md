# Repository Guidelines

## Project Structure & Module Organization
本仓库是 CS336 Assignment 1（Basics）的代码模板，核心实现位于 `cs336_basics/`，主要模块包括 `bpe.py`、`tokenizer.py`、`model.py`。  
测试位于 `tests/`，其中 `tests/adapters.py` 是你对接自己实现与测试框架的入口；`tests/fixtures/` 存放基准数据与权重文件。  
数据文件默认放在 `data/`（如 TinyStories、OWT 样本），说明文档与作业 PDF 在仓库根目录与 `doc/`。

## Build, Test, and Development Commands
- `uv sync`：根据 `pyproject.toml` 与 `uv.lock` 创建/同步开发环境与依赖。
- `uv run pytest`：运行全部单元测试。
- `uv run pytest tests/test_tokenizer.py -q`：仅运行单模块测试，便于快速迭代。
- `uv run pytest -k bpe`：按关键字筛选测试。
- `uv run python -c "import cs336_basics"`：快速验证包可被正确导入。

## Coding Style & Naming Conventions
使用 Python 3.10+，统一 4 空格缩进。函数、变量使用 `snake_case`，类名使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。  
建议为公共函数补充类型注解与简短 docstring，保持接口清晰。  
实现应尽量放在 `cs336_basics/*.py`，不要把核心逻辑写进测试文件；`tests/adapters.py` 只做“适配与调用”。

## Testing Guidelines
测试框架为 `pytest`（见 `pytest.ini`）。提交前至少运行 `pytest` 并确保新增改动覆盖对应测试。  
新增测试文件命名遵循 `tests/test_*.py`，测试函数使用 `test_*` 前缀。  
涉及数值计算（如 attention、optimizer）时，优先复用 `tests/fixtures/` 中样例，避免手写脆弱断言。

## Commit & Pull Request Guidelines
历史提交以简洁的 Conventional Commit 风格为主（如 `feat(tokenizer): ...`、`fix: ...`）。  
建议格式：`type(scope): summary`，`type` 常用 `feat`、`fix`、`refactor`、`test`。  
PR 需包含：变更目的、关键实现点、测试命令与结果；若修改行为或接口，请注明影响范围并关联 issue。

## Security & Configuration Tips
不要提交大体量原始数据、模型权重或密钥；本地实验产物应加入 `.gitignore`。  
下载数据时使用 `data/` 目录并保持文件名稳定，避免破坏测试与脚本约定路径。
