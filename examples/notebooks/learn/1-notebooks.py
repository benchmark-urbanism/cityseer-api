import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Computational Notebooks

    Computational notebooks provide an interactive environment for combining code, its results, and explanatory text. They are well suited to iterative tasks such as data analysis, because they let you test and document your work step by step. Immediate feedback and ease of experimentation make notebooks an accessible way to learn Python.

    > **Note:**
    > ## Learning Objectives
    >
    > - Understand what computational notebooks are and why they are useful for data analysis
    > - Open and run a marimo notebook
    > - Work with code and markdown cells
    > - Understand how marimo tracks variables and keeps cells up to date

    ## marimo

    These lessons use [marimo](https://marimo.io) notebooks. A marimo notebook is a plain Python file (this lesson is the file `1_notebooks.py`), which means it works cleanly with version control and can also be run as an ordinary Python script.

    To open a notebook for editing, run the following from the repository root:

    ```bash
    uv run marimo edit examples/class/1_notebooks.py
    ```

    This opens the notebook in your web browser, where you can run cells, edit code, and see results.

    ## Cells

    A notebook is made up of cells, of two kinds:

    - **Code cells** contain Python code. Outputs appear with the cell.
    - **Markdown cells** contain formatted text for documentation and context.

    ## Running Code

    To run a code cell, select it and press `Ctrl+Enter` (or `Cmd+Enter` on Mac), or click its play button. Here's an example:
    """)
    return


@app.cell
def _():
    2 + 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output appears with the cell.

    Another example:
    """)
    return


@app.cell
def _():
    x = 5
    print(x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implicit Output

    A handy feature of notebooks is that the value of the last expression in a code cell is automatically displayed as output:
    """)
    return


@app.cell
def _():
    # The value of 'answer' will be displayed
    answer = 42
    answer
    return (answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To display other values, or values that aren't on the last line of a cell, use the `print()` function:
    """)
    return


@app.cell
def _(answer):
    # Explicitly displaying values
    print(answer)  # This will show the value of 'answer'
    doubled = answer * 2
    doubled  # Automatically displayed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How marimo Tracks Variables

    A variable defined in one cell can be used in other cells. In the next cell we define `city_name`:
    """)
    return


@app.cell
def _():
    # Define a variable
    city_name = "Seville"
    return (city_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Other cells can then use it:
    """)
    return


@app.cell
def _(city_name):
    print("The selected city is:", city_name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Behind the scenes, marimo reads each cell to see which variables it defines and which it uses, and connects the cells into a *dataflow graph*. This has two practical consequences that distinguish marimo from traditional notebooks (such as Jupyter):

    1. **Cells re-run automatically.** If you change `city_name` to `"Barcelona"` in its defining cell and run it, every cell that uses `city_name` re-runs with the new value. You never end up with stale results from cells that were run in the wrong order, which is a common source of confusion in traditional notebooks.
    2. **A variable can be defined in only one cell.** If two cells both assigned to `city_name`, marimo would not know which value the other cells should use, so it reports an error. To fix such an error, either use a different variable name or combine the two definitions into one cell.

    Because of these rules, the position of a cell on the page does not determine when it runs; the variables connecting it to other cells do. Notebooks are nevertheless easier to read when they flow from top to bottom, so these lessons are arranged that way.

    A small dependency chain as an example:
    """)
    return


@app.cell
def _():
    base = 10
    base
    return (base,)


@app.cell
def _(base):
    tripled = base * 3
    tripled
    return (tripled,)


@app.cell
def _(tripled):
    total = tripled + 5
    total
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try changing `base` to a different number and running that cell: the `tripled` and `total` cells update automatically.

    > **Note:**
    > Variables whose names start with an underscore (e.g. `_result`) are *local* to their cell. They can be reused in other cells without conflict, which is useful for throwaway values such as plot axes.

    ## Adding Text

    Markdown cells use [Markdown](https://www.markdownguide.org/basic-syntax/), a lightweight markup language that uses simple characters for formatting: headings, bold or italic text, lists, and tables. In the marimo editor, create a markdown cell by adding a new cell and switching it to markdown mode (or start typing markdown and marimo will offer to convert it).

    ## Tips

    - **Troubleshooting errors:** Read the error message shown with the cell; marimo also highlights problems such as a variable defined in more than one cell.
    - **Saving:** Saving the notebook simply saves the `.py` file. Because it is plain Python, it diffs cleanly in git and can be shared like any other file.
    - **Running as a script:** A marimo notebook can also be executed directly, e.g. `uv run python examples/class/1_notebooks.py`.
    - **Getting help:** To see what a function does, run `help(function_name)` in a code cell.

    ### Exercises

    - Create a new code cell, assign the value `100` to a variable called `distance`, and print it.
    - Create a markdown cell with a heading and a short description of what the notebook is about.
    - Change the value of `base` above and observe which cells re-run.
    - Try defining `city_name` a second time in a new cell. What does marimo report? Delete the cell afterwards.

    ## Summary

    - Computational notebooks combine code, output, and explanatory text in an interactive document.
    - A marimo notebook is a plain Python file; open it with `uv run marimo edit <file>`.
    - Notebooks contain code cells (Python) and markdown cells (documentation).
    - marimo connects cells into a dataflow graph: a variable may be defined in only one cell, and dependent cells re-run automatically when their inputs change.
    - Underscore-prefixed variables are local to a cell.

    Next: [Python Basics](https://cityseer.benchmarkurbanism.com/start/2-basics) covers the fundamentals of the Python language.
    """)
    return


if __name__ == "__main__":
    app.run()
