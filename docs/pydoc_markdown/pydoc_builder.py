import sys
import pathlib

from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor
from pydoc_markdown.contrib.processors.filter import FilterProcessor

from custom_components.custom_renderer import CustomMarkdownRenderer
from custom_components.custom_processor import CustomProcessor


# add cityseer directory to sys path so that modules can resolve from build scripts
this_file_dir = pathlib.Path(__file__).parent
cityseer_package_root = pathlib.Path(this_file_dir.parent.parent)
sys.path.append(cityseer_package_root)

# docs directory for output
docs_dir = this_file_dir.parent / 'vitepress'

module_set = (
    ('cityseer.metrics.layers', 'metrics/layers.md'),
    ('cityseer.metrics.networks', 'metrics/networks.md'),
    ('cityseer.tools.graphs', 'tools/graphs.md'),
    ('cityseer.tools.mock', 'tools/mock.md'),
    ('cityseer.tools.plot', 'tools/plot.md')
)

for module_path, doc_path in module_set:
    pydocmd = PydocMarkdown(
        loaders=[PythonLoader(
            search_path=None,
            modules=[module_path],
            packages=None
        )],
        processors=[
            FilterProcessor(skip_empty_modules=True),
            CrossrefProcessor(),
            CustomProcessor()
        ],
        renderer=CustomMarkdownRenderer()
    )
    # create the path and output directories as needed
    out_file = pathlib.Path(docs_dir / doc_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # render
    pydocmd.renderer.set_filename(out_file)
    modules = pydocmd.load_modules()
    pydocmd.process(modules)
    for module in modules:
        # dump_module(module, sys.stdout)
        pydocmd.render([module])
