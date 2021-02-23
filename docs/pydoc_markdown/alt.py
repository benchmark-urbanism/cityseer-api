import sys
import pathlib

import docspec
import docspec_python

# add cityseer directory to sys path so that modules can resolve from build scripts
this_file_dir = pathlib.Path(__file__).parent
cityseer_package_root = pathlib.Path(this_file_dir.parent.parent)
sys.path.append(cityseer_package_root)

# docs directory for output
docs_dir = this_file_dir.parent / 'vitepress'

module_set = (
    #('cityseer.metrics.layers', 'metrics/layers.md'),
    #('cityseer.metrics.networks', 'metrics/networks.md'),
    #('cityseer.tools.graphs', 'tools/graphs.md'),
    #('cityseer.tools.mock', 'tools/mock.md'),
    ('cityseer.tools.plot', 'tools/plot.md'),
)

for module_name, doc_path in module_set:
    # process the module
    modules = docspec_python.load_python_modules(modules=[module_name])
    for module in modules:
        lines = []
        lines.append(f'# {module.name}\n')
        for member in module.members:
            # not yet processing module-level variables
            if isinstance(member, docspec.Data):
                continue
            # keep track of the arguments and their types
            arg_types_map = {}
            # process the signature
            if isinstance(member, docspec.Function):
                lines.append(f'### {member.name}\n\n<FuncSignature>\n<pre>\n')
                lines.append(f'{member.name}(')
                spacer = len(member.name) + 1
                for idx, arg in enumerate(member.args):
                    # add to the arg to types map
                    arg_types_map[arg.name] = arg.datatype
                    # if the argument type is KeywordRemainder then add symbols
                    arg_name = arg.name
                    if arg.type.name == 'KeywordRemainder':
                        arg_name = '**' + arg_name
                    # first argument is wedged against bracket
                    if idx == 0:
                        lines[-1] += arg_name
                    else:
                        lines.append(f'{" " * spacer}{arg_name}')
                    if arg.default_value is not None:
                        lines[-1] += f' = {arg.default_value}'
                    # if not the last argument, add a comma
                    if idx != len(member.args) - 1:
                        lines[-1] += ','
                lines[-1] += ')\n</pre>\n</FuncSignature>\n\n'
                if member.return_type is not None:
                    lines[-1] += f' -> {member.return_type}'
            # process the docstring
            headline_map = {
                'args': 'Arguments',
                'arguments': 'Arguments',
                'returns': 'Return',
                'example': 'Example'
            }
            if member.docstring is not None:
                # compact "wrapped" lines
                compacted_lines = []
                heading_mode = None
                param_name = None
                code_block = False
                param_description = []
                # iterate docstring lines
                splits = member.docstring.split('\n')
                for line in splits:
                    # empty lines
                    if line.strip() == '':
                        # reset heading_mode if necessary
                        if heading_mode is not None:
                            if heading_mode is 'Example':
                                compacted_lines.append(param_description)
                            else:
                                param_type = arg_types_map[param_name]
                                param_description = '\n'.join(param_description)
                                compacted_lines.append(f'<FuncElement name="{param_name}" type="{param_type}">\n'
                                                       f'{param_description}\n'
                                                       f'</FuncElement>\n')
                            heading_mode = None
                            param_description = []
                        compacted_lines.append('')
                        continue
                    # if in a code block,
                    if line.strip().startswith('```'):
                        code_block = not code_block
                    if code_block:
                        compacted_lines.append(line[4:])
                        continue
                    # watch for new heading_mode
                    if line.strip(' :').lower() in headline_map:
                        heading_mode = headline_map[line.strip(' :').lower()]
                        compacted_lines.append(f'<FuncHeading>{heading_mode}</FuncHeading>\n')
                        continue
                    # if first line of heading_mode
                    if line.startswith('    ') and line[4] != ' ':
                        # strip the characters and split at the colon
                        param_splits = line.strip().split(':')
                        # there should not be any type information, but split again by space to be sure discarded
                        param_name = param_splits.pop(0).split(' ')[0]
                        # in case more than one colon
                        param_description.append(':'.join(param_splits))
                        continue
                    # preserve linebreaks for bullets, tables, pictures, captions
                    if line.startswith('        ') and line.strip()[0] in ['*', '-', '|', '!', '_']:
                        # keep spacing after eight spaces so that lists and intentional spaces aren't affected
                        param_description.append(line[8:])
                        continue
                    # otherwise, wrap
                    if heading_mode and len(param_description):
                        param_description[-1] += ' ' + line.strip()
                    elif heading_mode:
                        param_description.append(line)
                    elif len(compacted_lines):
                        compacted_lines[-1] += ' ' + line.strip()
                    else:
                        compacted_lines.append(line)
                # add the docstring
                lines += compacted_lines

        # create the path and output directories as needed
        out_file = pathlib.Path(docs_dir / doc_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, mode='w') as out_file:
            for line in lines:
                print(line)
                out_file.write(line)
