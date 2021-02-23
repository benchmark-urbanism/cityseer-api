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

import html
import logging
import re
from typing import List

import docspec
from docspec import Argument
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from nr.databind.core import Field

logger = logging.getLogger(__name__)


class CustomMarkdownRenderer(MarkdownRenderer):
    """
    See MarkdownRenderer for overridable fields
    https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/pydoc-markdown/src/pydoc_markdown/contrib/renderers/markdown.py
    """

    #: If enabled, inserts anchors before Markdown headers to ensure that
    #: links to the header work. This is enabled by default.
    insert_header_anchors = Field(bool, default=False)

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
    render_toc = Field(bool, default=False)

    #: The title of the "Table of Contents" header.
    render_toc_title = Field(str, default='Table of Contents')

    #: Include the "def" keyword in the function signature. This is enabled
    #: by default.
    signature_with_def = Field(bool, default=False)

    #: Render the function signature as a code block. This includes the "def"
    #: keyword, the function name and its arguments. This is enabled by
    #: default.
    signature_code_block = Field(bool, default=True)

    #: Allows you to override how the "view source" link is rendered into the Markdown
    #: file if a #source_linker is configured. The default is `[[view_source]]({url})`.
    source_format = Field(str, default='[[view_source]]({url})')

    #: Escape html in docstring. Default to False.
    escape_html_in_docstring = Field(bool, default=False)

    #: Conforms to Docusaurus header format.
    render_module_header_template = Field(str, default='')

    #: Fixed header levels by API object type.
    header_level_by_type = Field({int}, default={
        'Module': 1,
        'Class': 2,
        'Method': 3,
        'Function': 3,
        'Data': 3,
    })

    ### ADDED
    def set_filename(self, path):
        self.filename = path

    ### MODIFIED
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
        # go-ahead and add the first param
        # pad the parameters
        spaces = len(parts[0])
        arg_list_str = self._format_arglist(func)
        splits = arg_list_str.split(',')
        # gather the params
        pad = ''
        joiner = ','
        # split over multiple lines if more than x params or if more than y characters
        if len(splits) > 1 or len(arg_list_str) > 40:
            pad = ' ' * spaces
            joiner = ',\n'
        # insert the first param without spacing
        params = [splits[0]]
        # iterate and pad the splits
        for idx, split in enumerate(splits[1:]):
            params.append(f'{pad}{split}')
        # rejoin and concat
        parts.append(f'({joiner.join(params)})')
        if func.return_type:
            parts.append(f' -> {func.return_type}')
        # TODO: adding FuncSignature tags
        # pre tag prevents stripping of spaces or newlines
        result = '<FuncSignature>\n<pre>\n'
        result += ''.join(parts)
        result += '\n</pre>\n</FuncSignature>\n\n'
        if add_method_bar and self._is_method(func):
            result = '\n'.join(' | ' + l for l in result.split('\n'))
        return result

    ### TEMPORARY UNTIL docspec-python MERGE - THEN THIS CAN BE REMOVED...
    def _format_arglist(self, func: docspec.Function) -> str:
        args = func.args[:]
        if self._is_method(func) and args and args[0].name == 'self':
            args.pop(0)
        # TODO: temporarily recreating docspec-python format_arglist function here until / if Pull request is merged.
        result = []
        for arg in args:
            if arg.type == Argument.Type.KeywordOnly and '*' not in result:
                result.append('*')
            parts = [arg.name]
            if arg.default_value:
                parts.append(' = ')
            if arg.default_value:
                parts.append(arg.default_value)
            if arg.type == Argument.Type.PositionalRemainder:
                parts.insert(0, '*')
            elif arg.type == Argument.Type.KeywordRemainder:
                parts.insert(0, '**')
            result.append(''.join(parts))

        return ', '.join(result)


    ### MODIFIED
    def _render_signature_block(self, fp, obj):
        if self.classdef_code_block and isinstance(obj, docspec.Class):
            code = self._format_classdef_signature(obj)
        elif self.signature_code_block and isinstance(obj, docspec.Function):
            code = self._format_function_signature(obj)
        elif self.data_code_block and isinstance(obj, docspec.Data):
            code = self._format_data_signature(obj)
        else:
            return
        # TODO: removed code ticks
        fp.write(code)

    ### MODIFIED
    def _render_object(self, fp, level, obj):
        if not isinstance(obj, docspec.Module) or self.render_module_header:
            self._render_header(fp, level, obj)
        url = self.source_linker.get_source_url(obj) if self.source_linker else None
        source_string = self.source_format.replace('{url}', str(url)) if url else None
        if source_string and self.source_position == 'before signature':
            fp.write(source_string + '\n\n')
        self._render_signature_block(fp, obj)
        if source_string and self.source_position == 'after signature':
            fp.write(source_string + '\n\n')
        if obj.docstring:
            docstring = html.escape(obj.docstring) if self.escape_html_in_docstring else obj.docstring
            lines = docstring.split('\n')
            if self.docstrings_as_blockquote:
                lines = ['> ' + x for x in lines]
            # TODO: added FuncHeading tags
            # Use Regex to find headings
            for idx, line in enumerate(lines):
                if re.match('\*\*.*\*\*:', line) is not None:
                    line = line.strip('*:')
                    lines[idx] = f'<FuncHeading>\n\n{line}\n\n</FuncHeading>\n'
                else:
                    lines[idx] = line.strip()
            fp.write('\n'.join(lines))
            fp.write('\n\n')