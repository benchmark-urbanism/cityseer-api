import sys
from pathlib import Path

from docspec import dump_module
from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor
from pydoc_markdown.contrib.processors.filter import FilterProcessor

from docs.pydoc_markdown.custom_renderer import CustomizedMarkdownRenderer
from docs.pydoc_markdown.custom_processor import CustomProcessor

package_root = Path(__file__).parent.parent.parent
p = str(Path(package_root / 'cityseer/util'))

pydocmd = PydocMarkdown(
    loaders=[PythonLoader(
        search_path=[p],
        modules=None,
        packages=None
    )],
    processors=[
        FilterProcessor(skip_empty_modules=True),
        CrossrefProcessor(),
        CustomProcessor()
    ],
    renderer=CustomizedMarkdownRenderer()
)

# RS = RenderSession(config=None,
#                    render_toc=True,
#                    search_path=[p],
#                    modules=None,
#                    packages=None,
#                    py2=False)
#
# pydocmd = RS.load()

modules = pydocmd.load_modules()
pydocmd.process(modules)
for module in modules:
    dump_module(module, sys.stdout)
    pydocmd.render([module])


print('here')
