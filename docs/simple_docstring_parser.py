"""
Uses docspec to parse docstrings to markdown.

Intended for use with static site generators where further linting / linking / styling is done downstream.

Loosely based on Numpy-style docstrings.

Automatically infers types from signature typehints. Explicitly documented types are NOT supported in docstrings.
"""
import pathlib
import sys

import docspec
import docspec_python
import yaml

root_dir = pathlib.Path(__file__).parent

# load the config
file_names = [
    '.simple_docstring_parser.yaml',
    '.simple_docstring_parser.yml'
]
yam = None
for file_name in file_names:
    file_path = pathlib.Path(root_dir / file_name)
    if file_path.exists():
        yam = yaml.load(open(file_path), Loader=yaml.SafeLoader)
        break
if yam is None:
    raise ValueError(f'Unable to find valid configuration file. This file should be placed in the project root'
                     f'directory and should be name either {" or ".join([f_n for f_n in file_names])}')

# check and add package root path if spec'd in config
# this should only be necessary if the script is placed somewhere other than the package root
if 'package_root_relative_path' in yam:
    package_path = pathlib.Path(root_dir, yam['package_root_relative_path'])
    print(package_path)
    sys.path.append(package_path)

# prepare the defaults
config = {
    'frontmatter_template': None,
    'module_name_template': '# {module_name}\n\n',
    'toc_template': None,
    'function_name_template': '\n\n## {function_name}\n\n',
    'class_name_template': '\n\n## **class** {class_name}\n\n',
    'class_property_template': '\n\n#### {prop_name}\n\n',
    'signature_template': '\n\n```py\n{signature}\n```\n\n',
    'heading_template': '\n\n#### {heading}\n\n',
    'param_template': '\n\n**{name}** _{type}_: {description}\n\n',
    'return_template': '\n\n**{name}**: {description}\n\n'
}
# check whether the defaults are overriden from the config
for def_key in config.keys():
    if def_key in yam:
        config[def_key] = yam[def_key]
# parse the module / doc path pairs
if not 'module_map' in yam:
    raise KeyError('The configuration file requires a dictionary mapping modules to output paths for markdown.')

line_break_chars = ['-', '_', '!', '|', '>', ':']


def is_property(mem):
    if mem.decorations is not None:
        for dec in mem.decorations:
            if dec.name == 'property':
                return True
    return False


def is_setter(mem):
    if mem.decorations is not None:
        for dec in mem.decorations:
            if 'setter' in dec.name:
                return True
    return False


def extract_text_block(splits_enum, splits, indented_block=False, is_hint_block=False):
    """
    Parses a block of text and decides whether or not to wrap.
    Return if iters finish or on end of indentation (optional) or on start of new heading
    """
    block = []
    while True:
        # feed
        lookahead_idx, next_line = next(splits_enum)
        # return if indented block and next is not indented (but don't get tripped up with empty lines)
        if indented_block and not next_line.startswith(' ') and not next_line.strip() == '':
            return lookahead_idx, next_line, '\n'.join(block)
        # return if the next next-line would be a new heading
        elif lookahead_idx < len(splits) and splits[lookahead_idx].startswith('---'):
            return lookahead_idx, next_line, '\n'.join(block)
        # return if inside a hint block and the end of the hint block has been encountered
        elif is_hint_block and next_line.strip().startswith(':::'):
            return lookahead_idx, next_line, '\n'.join(block)
        # be careful with stripping content for lines with intentional breaks, e.g. indented bullets...
        # if parsing indented blocks, strip the first four spaces
        if indented_block:
            next_line = next_line[4:]
        # code blocks
        if next_line.strip().startswith('```'):
            code_block = next_line.strip() + '\n'
            while True:
                lookahead_idx, next_line = next(splits_enum)
                if indented_block:
                    next_line = next_line[4:]
                code_block += next_line + '\n'
                if next_line.startswith('```'):
                    break
            block.append(code_block)
        # tip blocks
        elif next_line.strip().startswith(':::'):
            hint_in = '\n' + next_line.strip() + '\n\n'
            # unpacks hint block
            lookahead_idx, next_line, hint_block = extract_text_block(splits_enum,
                                                                      splits,
                                                                      indented_block=indented_block,
                                                                      is_hint_block=True)
            # next line will be closing characters, i.e. ':::', insert manually to add newline
            block.append(hint_in + hint_block + '\n:::')
        # if no block content exists yet
        elif not len(block):
            block.append(next_line)
        # keep blank lines
        elif next_line.strip() == '':
            block.append('')
        # don't wrap if the previous line is blank
        elif block[-1] == '':
            block.append(next_line)
        # don't wrap if the line starts with a bullet point, picture, or table character
        elif next_line.strip()[0] in line_break_chars:
            block.append(next_line)
        # or if the previous line ends with a bullet point, picture, or table character
        elif block[-1].strip()[-1] in line_break_chars:
            block.append(next_line)
        # otherwise wrap
        else:
            # should be safe to strip text when wrapping
            block[-1] += ' ' + next_line.strip()
        # return if iters exhausted
        if lookahead_idx == len(splits):
            return lookahead_idx, next_line, '\n'.join(block)


def process_member(member, lines, class_name=None):
    # this method only processes functions and classes
    if not isinstance(member, (docspec.Function, docspec.Class)):
        return
    # don't process private members
    if (member.name.startswith('_') and not member.name == '__init__') or is_setter(member):
        return
    # keep track of the arguments and their types for automatically building function parameters later-on
    arg_types_map = {}
    # escape underscores in class / method / function names
    member_name = member.name.replace('_', '\_')
    if class_name is not None:
        class_name_esc = class_name.replace('_', '\_')
        # if a class definition use the class template
        if isinstance(member, docspec.Class):
            # when the class is passed-in directly its name is captured in the member_name
            lines.append(config['class_name_template'].format(class_name=class_name_esc))
        # if the class __init__, then display the class name and .__init__
        elif class_name and member.name == '__init__':
            lines.append(config['function_name_template'].format(function_name=f'{class_name_esc}'))
        # if a class property
        elif class_name is not None and is_property(member):
            lines.append(config['class_property_template'].format(prop_name=f'{class_name_esc}.{member_name}'))
        # if a class method
        elif class_name is not None:
            lines.append(config['function_name_template'].format(function_name=f'{class_name_esc}.{member_name}'))
    # otherwise a function
    else:
        lines.append(config['function_name_template'].format(function_name=member_name))
    # process the member's signature if a method or a function - classes won't have args
    if hasattr(member, 'args') and not is_property(member):
        # prepare the signature string - use member.name instead of escaped versions
        if class_name is not None and member.name == '__init__':
            signature = f'{class_name}('
        elif class_name is not None:
            signature = f'{class_name}.{member.name}('
        else:
            signature = f'{member.name}('
        # the spacer is used for lining up wrapped lines
        spacer = len(signature)
        # unpack the arguments and add
        for idx, arg in enumerate(member.args):
            # ignore self parameter
            if arg.name == 'self':
                continue
            # param name
            param_name = arg.name
            # add to the arg_types_map map using the function / method name and param name
            arg_types_map[param_name] = arg.datatype
            # if the argument type is KeywordRemainder then add the symbols
            if arg.type.name == 'KeywordRemainder':
                param_name = '**' + param_name
            # first argument is wedged against bracket
            # except for classes where self parameters are ignored and second argument is wedged
            if idx == 0 or class_name is not None and idx == 1:
                signature += param_name
            # other arguments start on a new line
            else:
                signature += f'{" " * spacer}{param_name}'
            # add default values where present
            if arg.default_value is not None:
                signature += f'={arg.default_value}'
            # if not the last argument, add a comma
            if idx != len(member.args) - 1:
                signature += ',\n'
        # close the signature
        signature += ')'
        # add the return type if present
        if member.return_type is not None:
            signature += f'\n{" " * spacer}-> {member.return_type}'
        # set into the template
        signature = config['signature_template'].format(signature=signature)
        lines.append(signature)
    # process the docstring
    if member.docstring is not None:
        # split the docstring at new lines
        splits = member.docstring.split('\n')
        # iter the docstring with a lookahead index
        splits_enum = enumerate(splits, start=1)
        try:
            # skip and go straight to headings if no introductory text
            if len(splits) > 1 and splits[1].startswith('---'):
                lookahead_idx, next_line = next(splits_enum)
            # otherwise, look for introductory text
            else:
                lookahead_idx, next_line, text_block = extract_text_block(splits_enum, splits)
                if len(text_block):
                    lines.append(text_block)
            # look for headings
            while lookahead_idx < len(splits):
                # break if not a heading
                if not splits[lookahead_idx].startswith('---'):
                    raise ValueError('Parser out of lockstep with headings.')
                heading = next_line.strip()
                lines.append(config['heading_template'].format(heading=heading))
                # skip the underscore line
                next(splits_enum)
                # if not param-type headings - just extract the text blocks
                if heading not in ['Parameters', 'Returns', 'Yields', 'Raises']:
                    lookahead_idx, next_line, text_block = extract_text_block(splits_enum, splits)
                    if len(text_block):
                        lines.append(text_block)
                # otherwise iterate the parameters and their indented arguments
                else:
                    # initial prime to move from heading to parameter name
                    lookahead_idx, next_line = next(splits_enum)
                    # Iterate nested parameters
                    while True:
                        # this parser doesn't process typehints, use typehints in function declarations instead
                        if ' ' in next_line.strip() or ':' in next_line.strip():
                            raise ValueError('Parser does not support types in docstrings. Use type-hints instead.')
                        # extract the parameter name
                        param_name = next_line.strip()
                        # process the indented parameter description
                        lookahead_idx, next_line, param_description = extract_text_block(splits_enum,
                                                                                         splits,
                                                                                         indented_block=True)
                        # only include type information for Parameters
                        if heading == 'Parameters':
                            param_type = arg_types_map[param_name]
                            param = config['param_template'].format(name=param_name,
                                                                    type=param_type,
                                                                    description=param_description)
                        else:
                            param = config['return_template'].format(name=param_name,
                                                                     description=param_description)
                        lines.append(param)
                        # break if a new heading found
                        if lookahead_idx == len(splits) or splits[lookahead_idx].startswith('---'):
                            break
        # catch exhausted enum
        except StopIteration:
            pass


for module_name, doc_path in yam['module_map'].items():
    # process the module
    modules = docspec_python.load_python_modules(modules=[module_name])
    for module in modules:
        lines = []
        # frontmatter
        if config['frontmatter_template'] is not None:
            lines.append(config['frontmatter_template'])
        # module name
        lines.append(config['module_name_template'].format(module_name=module_name).replace('_', '\_'))
        # module docstring
        if module.docstring is not None:
            lines.append(module.docstring.strip().replace('\n', ' '))
        if config['toc_template'] is not None:
            lines.append(config['toc_template'])
        # iterate the module's members
        for member in module.members:
            # ignores module-level variables
            if isinstance(member, docspec.Data):
                continue
            # process functions
            elif isinstance(member, docspec.Function):
                process_member(member, lines)
            # process classes and nested methods
            elif isinstance(member, docspec.Class):
                class_name = member.name
                process_member(member, lines, class_name)
                for nested_member in member.members:
                    process_member(nested_member, lines, class_name)
        # create the path and output directories as needed
        out_file = pathlib.Path(doc_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # write!
        with open(out_file, mode='w') as out_file:
            for line in lines:
                out_file.write(line)
