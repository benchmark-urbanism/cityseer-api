# cityseer examples

Worked examples, tutorials, and datasets for [cityseer](https://cityseer.benchmarkurbanism.com). The rendered documentation for this content lives on the [examples pages](https://cityseer.benchmarkurbanism.com/examples) of the main site.

All notebooks are [marimo](https://marimo.io) notebooks: plain Python files that open interactively, run as scripts, and diff cleanly in version control. Nothing ties the code to marimo; you can create a notebook in your own preferred environment and copy the cells across.

## Layout

- `recipes/` — task-oriented notebooks: network preparation, centrality, accessibility, statistics, visibility, and continuity. Start with `recipes/quickstart.py`.
- `class/` — Python 101: six lessons covering notebooks, Python basics, spatial data, GeoPandas, urban analytics, and data science.
- `data/` — real-world datasets used by the recipes; sources and licensing on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
- `cases/` — longer-form case studies.

## Running

From the repository root:

```bash
uv sync --group examples
uv run marimo edit examples/recipes/quickstart.py
```

Each notebook is also a runnable script: `uv run python examples/recipes/quickstart.py`.

## License

The examples and lesson content in this directory are licensed under CC BY-NC-SA 4.0 (see `LICENSE`); the cityseer library itself is AGPL-3.0. Datasets retain their original source licenses as documented on the datasets page.
