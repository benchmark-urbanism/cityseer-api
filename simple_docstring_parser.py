import pathlib

import docspec
import docspec_python
import yaml

root_dir = pathlib.Path(__file__).parent

# load the config
file_names = [
    '.simple_docstring_parser.yaml',
    '.simple_docstring_parser.yml'
]
config = None
for file_name in file_names:
    file_path = pathlib.Path(root_dir / file_name)
    if file_path.exists():
        config = yaml.load(open(file_path), Loader=yaml.SafeLoader)
        break
if config is None:
    raise ValueError(f'Unable to find valid configuration file. This file should be placed in the project root'
                     f'directory and should be name either {" or ".join([f_n for f_n in file_names])}')

# use double breaks - linter will strip out redundancies
module_name_template = '# {module_name}\n\n'
if 'module_name_template' in config:
    module_name_template = config['module_name_template']

function_name_template = '\n\n### {function_name}\n\n'
if 'function_name_template' in config:
    function_name_template = config['function_name_template']

signature_template = '\n\n```py\n{signature}\n```\n\n'
if 'signature_template' in config:
    signature_template = config['signature_template']

heading_template = '\n\n## {heading}\n\n'
if 'heading_template' in config:
    heading_template = config['heading_template']

param_template = '\n\n**{name}** _{type}_: {description}\n\n'
if 'param_template' in config:
    param_template = config['param_template']

return_template = '\n\n**{name}**: {description}\n\n'
if 'return_template' in config:
    return_template = config['return_template']

# parse the module / doc path pairs
if not 'module_map' in config:
    raise KeyError('The configuration file requires a dictionary mapping modules to output paths for markdown.')
for module_name, doc_path in config['module_map'].items():
    # process the module
    modules = docspec_python.load_python_modules(modules=[module_name])
    for module in modules:

        # first-off, add the module name
        lines = [module_name_template.format(module_name=module_name)]

        # iterate the module's members
        for member in module.members:

            # ignores module-level variables
            if isinstance(member, docspec.Data):
                continue

            # process the signature
            if isinstance(member, docspec.Function) and not member.name.startswith('_'):
                # keep track of the arguments and their types for automatically building function parameters later-on
                arg_types_map = {}
                # the name as header, followed by Vue component and function name
                function_name = member.name
                lines.append(function_name_template.format(function_name=function_name))
                # prepare the signature string
                signature = f'{function_name}('
                # the spacer is used for lining up wrapped lines
                spacer = len(signature)
                # unpack the arguments and add
                for idx, arg in enumerate(member.args):
                    # param name
                    param_name = arg.name
                    # add to the arg_types_map map using the function / method name and param name
                    arg_types_map[param_name] = arg.datatype
                    # if the argument type is KeywordRemainder then add the symbols
                    if arg.type.name == 'KeywordRemainder':
                        param_name = '**' + param_name
                    # first argument is wedged against bracket
                    if idx == 0:
                        signature += param_name
                    # other arguments start on a new line
                    else:
                        signature += f'{" " * spacer}{param_name}'
                    # add default values where present
                    if arg.default_value is not None:
                        signature += f' = {arg.default_value}'
                    # if not the last argument, add a comma
                    if idx != len(member.args) - 1:
                        signature += ',\n'
                # close the signature
                signature += ')'
                # add the return type if present
                if member.return_type is not None:
                    signature += f' -> {member.return_type}'
                # set into the template
                signature = signature_template.format(signature=signature)
                lines.append(signature)

                # process the docstring
                if member.docstring is not None:
                    # split the docstring at new lines
                    splits = member.docstring.split('\n')
                    # iter the docstring with a lookahead index
                    splits_enum = enumerate(splits, start=1)
                    while True:
                        try:
                            lookahead_idx, next_line = next(splits_enum)
                            # empty lines
                            if next_line.strip() == '':
                                lines.append('')
                                continue
                            # code blocks
                            if next_line.strip().startswith('```'):
                                code = next_line.strip() + '\n'
                                while True:
                                    lookahead_idx, next_line = next(splits_enum)
                                    code += next_line + '\n'
                                    if next_line.startswith('```'):
                                        break
                                lines.append(code)
                                continue
                            # look-ahead to find headings
                            if splits[lookahead_idx].startswith('---'):
                                heading = next_line.strip()
                                lines.append(heading_template.format(heading=heading))
                                # skip the next line
                                next(splits_enum)
                                # don't extract parameters unless a parameter type heading
                                if heading not in ['Parameters', 'Returns', 'Yields', 'Raises']:
                                    continue
                                # extract parameters and their descriptions
                                while True:
                                    lookahead_idx, next_line = next(splits_enum)
                                    # break if the next line is empty
                                    if next_line.strip() == '':
                                        lines.append('')
                                        break
                                    # parameters should not start with spaces and should have no internal spaces
                                    if not next_line.startswith('    '):
                                        if ' ' in next_line.strip() or ':' in next_line.strip():
                                            raise ValueError('This parser does not support types in docstrings. '
                                                             'Use type-hints instead.')
                                        # this line should be the parameter name
                                        param_name = next_line.strip()
                                        # the next line should be a description and should start with four spaces
                                        lookahead_idx, next_line = next(splits_enum)
                                        if not next_line.startswith('    '):
                                            raise ValueError('Parameter missing a description.')
                                        param_description = next_line.strip()
                                        # wrap any subsequent lines
                                        while True:
                                            # break once encountering another parameter or an empty line
                                            if splits[lookahead_idx] == '' \
                                                    or not splits[lookahead_idx].startswith('    '):
                                                break
                                            lookahead_idx, next_line = next(splits_enum)
                                            param_description += ' ' + next_line.strip()
                                        # wrap-up param
                                        if heading == 'Parameters':
                                            param_type = arg_types_map[param_name]
                                            param = param_template.format(name=param_name,
                                                                          type=param_type,
                                                                          description=param_description)
                                        else:
                                            param = return_template.format(name=param_name,
                                                                          description=param_description)
                                        lines.append(param)
                            # otherwise, just return the line
                            lines.append(next_line)
                            continue
                        # break when iter exhausted
                        except IndexError:
                            break

        # create the path and output directories as needed
        out_file = pathlib.Path(doc_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # write!
        with open(out_file, mode='w') as out_file:
            for line in lines:
                out_file.write(line)
