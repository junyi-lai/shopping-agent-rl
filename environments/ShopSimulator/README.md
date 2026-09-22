# ShopSimulator

Embedded [ShopSimulator](https://arxiv.org/pdf/2601.18225) Environment v2.1: the
frozen product corpus (23,421 products), the multi-field BM25 index builder, the
leased-session HTTP service (`/api/shop_agent`) and its unit tests. The upstream
source commit this snapshot was taken from is recorded in
[`EMBEDDED_SOURCE.json`](EMBEDDED_SOURCE.json).

The environment is a **pure simulator**: it renders the shop pages, keeps the
transactional session state and returns raw facts (purchase, target product,
candidates, variant price, termination reason). It never scores a trajectory —
termination is liveness-only (max steps, exact repeats, no progress) and the
reward lives in `src/shopping_agent/reward/`.

Do not install or start this directory by hand. From the repository root:

```bash
python scripts/00_environment.py setup                # 依赖、商品语料、索引、veRL 补丁、环境自检
python scripts/00_environment.py start --background   # 启动环境服务(默认 5700)
```

`setup` also runs the unit tests below with the environment's own Python 3.10
interpreter:

```bash
# 在 environments/ShopSimulator/shop_env 下执行
../.venv-shopsim/bin/python -m unittest discover -s tests -t .
```

Generated product JSON, search indexes, the virtual environment and logs are not
committed.
