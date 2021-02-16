import shutil
from pathlib import Path

from pdoc import pdoc, render

here = Path(__file__).parent
# out = here / '..' / 'vitepress'
# shutil.rmtree(out)

# Render pdoc's documentation into docs/api...
render.configure(template_directory=Path(here / 'templates'),
                 docformat=None,
                 edit_url_map=None)
pdoc(Path(here / '../..' / 'cityseer/util/plot.py'),
     output_directory=Path(here / '..' / 'vitepress/'),
     format='html')

# ...and rename the .html files to .md so that mkdocs picks them up!
# for f in out.glob('**/*.html'):
#     f.rename(f.with_suffix('.md'))