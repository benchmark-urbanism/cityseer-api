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

import re
from typing import List

import docspec
from pydoc_markdown.contrib.processors.google import GoogleProcessor
from pydoc_markdown.contrib.processors.sphinx import generate_sections_markdown


class CustomProcessor(GoogleProcessor):
    """
    Overrides the default Google Processor per:
    https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/pydoc-markdown/src/pydoc_markdown/contrib/processors/google.py
    """

    def _process(self, node: docspec.ApiObject):
        if not node.docstring:
            return

        lines = []
        current_lines: List[str] = []
        in_codeblock = False
        keyword = None

        def _commit():
            if keyword:
                generate_sections_markdown(lines, {keyword: current_lines})
            else:
                lines.extend(current_lines)
            current_lines.clear()

        # TODO: modified so that trailing lines are wrapped and appended to prior lines
        compacted_lines = []
        for line in node.docstring.split('\n'):
            # Retain spacing (markdown will be linted anyway)
            if line.strip() == '':
                compacted_lines.append('')
            # don't mess with bullet points, numbering, or tables
            elif line.strip()[0] in ['*', '-', '|'] or line.strip()[0].isnumeric():
                compacted_lines.append(line)
            # wrap trailing lines
            elif line.startswith('        '):
                compacted_lines[-1] = compacted_lines[-1] + ' ' + line.strip()
            else:
                compacted_lines.append(line)

        for line in compacted_lines:
            if line.startswith("```"):
                in_codeblock = not in_codeblock
                current_lines.append(line)
                continue
            if in_codeblock:
                current_lines.append(line)
                continue
            if line.strip() in self._keywords_map:
                _commit()
                keyword = self._keywords_map[line.strip()]
                continue
            if keyword is None:
                lines.append(line)
                continue
            for param_re in self._param_res:
                param_match = param_re.match(line.strip())
                if param_match:
                    if 'type' in param_match.groupdict():
                        # TODO: MODIFIED
                        current_lines.append(
                            '<FuncElement name="{param}" type="{type}">\n\n{desc}\n\n</FuncElement>\n'.format(
                                **param_match.groupdict()))
                    else:
                        # TODO: MODIFIED
                        current_lines.append(
                            '<FuncElement name="{param}">\n\n{desc}\n\n</FuncElement>\n'.format(**param_match.groupdict()))
                    break

            if not param_match:
                current_lines.append('  {line}'.format(line=line))

        _commit()
        node.docstring = '\n'.join(lines)
