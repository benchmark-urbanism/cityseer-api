# -*- coding: utf8 -*-
# Copyright (c) 2019 Niklas Rosenstein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import logging
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from nr.databind.core import Field, Struct
from nr.interface import implements, override
from pydoc_markdown.interfaces import Renderer
from typing import cast, List, TextIO
import docspec
import io
import sys

logger = logging.getLogger(__name__)


class CustomizedMarkdownRenderer(MarkdownRenderer):
    """
    See MarkdownRenderer for overridable fields
    https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/pydoc-markdown/src/pydoc_markdown/contrib/renderers/markdown.py
    """

    #: Render names in headers as code (using backticks or `<code>` tags,
    #: depending on #html_headers). This is enabled by default.
    code_headers = Field(bool, default=False)

    #: Generate descriptive class titles by adding the word "Objects" after
    #: the class name. This is enabled by default.
    descriptive_class_title = Field(bool, default=False)

    #: Generate descriptivie module titles by adding the word "Module" before
    #: the module name. This is enabled by default.
    descriptive_module_title = Field(bool, default=False)

    #: Add the class name as a prefix to method names. This class name is
    #: also rendered as code if #code_headers is enabled. This is enabled
    #: by default.
    add_method_class_prefix = Field(bool, default=True)

    #: Render a table of contents at the beginning of the file.
    render_toc = Field(bool, default=True)

    #: The title of the "Table of Contents" header.
    render_toc_title = Field(str, default='Table of Contents')

    #: Allows you to override how the "view source" link is rendered into the Markdown
    #: file if a #source_linker is configured. The default is `[[view_source]]({url})`.
    source_format = Field(str, default='[[view_source]]({url})')

    #: Escape html in docstring. Default to False.
    escape_html_in_docstring = Field(bool, default=False)

    #: Conforms to Docusaurus header format.
    render_module_header_template = Field(str, default=(
        '---\n'
        'sidebar_label: {relative_module_name}\n'
        'title: {module_name}\n'
        '---\n\n'
    ))

    def _format_function_signature(self,
                                   func: docspec.Function,
                                   override_name: str = None,
                                   add_method_bar: bool = True) -> str:
        parts: List[str] = []
        if self.signature_with_decorators:
            parts += self._format_decorations(func.decorations)
        if self.signature_python_help_style and not self._is_method(func):
            parts.append('{} = '.format(func.path()))
        parts += [x + ' ' for x in func.modifiers or []]
        if self.signature_with_def:
            parts.append('def ')
        if self.signature_class_prefix and self._is_method(func):
            parent = self._get_parent(func)
            assert parent, func
            parts.append(parent.name + '.')
        parts.append((override_name or func.name))
        parts.append('(' + self._format_arglist(func) + ')')
        if func.return_type:
            parts.append(' -> {}'.format(func.return_type))
        result = 'THIS IS A FUNCTION #####'
        result += ''.join(parts)
        if add_method_bar and self._is_method(func):
            result = '\n'.join(' | ' + l for l in result.split('\n'))
        return result

@implements(Renderer)
class CustomRenderer(Struct):
    markdown = Field(CustomizedMarkdownRenderer, default=CustomizedMarkdownRenderer)
    #: The path where the docs content is. Defaults "docs" folder.
    docs_base_path = Field(str, default='docs')
    #: The output path inside the docs_base_path folder, used to output the module reference.
    relative_output_path = Field(str, default='reference')

    @override
    def render(self, modules: List[docspec.Module]) -> None:
      self.render_to_stream(modules, sys.stdout)
