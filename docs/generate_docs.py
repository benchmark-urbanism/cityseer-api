from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import docstring_parser
from dominate import tags
from pdoc import doc, render


def gen_param_set(param_name: str, param_type: str | None, param_description: str) -> str:
    """Add a parameter set."""
    if not param_name:
        param_name = ""
    if param_type is None:
        param_type = "None"
    elem_desc_frag = tags.div(cls="desc")
    elem_desc_frag += strip_markdown(text=param_description)
    return (
        "\n"
        + tags.div(
            tags.div(
                tags.div(param_name, cls="name"),
                tags.div(param_type, cls="type"),
                cls="def",
            ),
            elem_desc_frag,
            cls="param-set",
        ).render()
        + "\n"
    )


def weld_candidate(text_a: str, text_b: str) -> bool:
    """Determine whether two strings can be merged into a single line."""
    if not text_a or text_a == "":
        return False
    if not text_b or text_b == "":
        return False
    for char in ["|", ">"]:
        if text_a.strip().endswith(char):
            return False
    for char in ["|", "!", "<", "-", "*"]:
        if text_b.strip().startswith(char):
            return False
    return True


def strip_markdown(text: str) -> str:
    """Add a markdown text block."""
    content_str = ""
    if text:
        content_str = text.strip()
    splits = content_str.split("\n")
    code_padding = None
    code_block = False
    other_block = False
    cleaned_text = "\n\n"
    for next_line in splits:
        # code blocks
        if "```" in next_line:
            if code_block is False:
                code_block = True
                code_padding = next_line.index("```")
                cleaned_text += f"\n{next_line[code_padding:]}"
            else:
                cleaned_text += f"\n{next_line[code_padding:]}\n"
                code_block = False
                code_padding = None
        elif code_block:
            cleaned_text += f"\n{next_line[code_padding:]}"
        # double breaks
        elif next_line == "":
            cleaned_text += "\n\n"
        # admonitions
        elif next_line.startswith(":::"):
            if not other_block:
                other_block = True
                cleaned_text += f"\n{next_line.strip()}"
            else:
                other_block = False
                cleaned_text += f"\n{next_line.strip()}"
        elif other_block:
            cleaned_text += f"\n{next_line.strip()}"
        # tables
        elif next_line.strip().startswith("|") and next_line.strip().endswith("|"):
            cleaned_text += f"\n{next_line.strip()}"
        # otherwise weld if possible
        elif weld_candidate(cleaned_text, next_line):
            cleaned_text += f" {next_line.strip()}"
        else:
            cleaned_text += f"\n{next_line.strip()}"
    if code_block or other_block:
        raise ValueError(f"Unclosed code block or admonition encountered for content: \n{cleaned_text}")
    cleaned_text = cleaned_text.replace("\n\n\n", "\n\n")
    return cleaned_text


def custom_process_docstring(doc_str: str) -> str:
    """Process a docstring."""
    doc_str_frag: str = ""
    parsed_doc_str = docstring_parser.parse(doc_str)
    if parsed_doc_str.short_description is not None:
        desc = parsed_doc_str.short_description
        if parsed_doc_str.long_description is not None:
            desc += f"\n{parsed_doc_str.long_description}"
        doc_str_frag += strip_markdown(text=desc)  # type: ignore
    if parsed_doc_str.params and len(parsed_doc_str.params):
        doc_str_frag += "\n### Parameters"
        for param in parsed_doc_str.params:
            param_name = param.arg_name
            if "kwargs" in param_name:
                param_name = param_name.lstrip("**")
                param_name = f"**{param_name}"
            doc_str_frag += gen_param_set(
                param_name=param_name,
                param_type=param.type_name,
                param_description=param.description,  # type: ignore
            )
    # track types parsed from return docstrings
    return_types_in_docstring: list[str] = []
    if parsed_doc_str.many_returns and len(parsed_doc_str.many_returns):
        doc_str_frag += "\n### Returns"
        for doc_str_return in parsed_doc_str.many_returns:
            if doc_str_return.type_name in [None, "None"]:
                param_type = None
            else:
                param_type = doc_str_return.type_name
                return_types_in_docstring.append(param_type)  # type: ignore
            # add fragment
            doc_str_frag += gen_param_set(
                param_name=doc_str_return.return_name,  # type: ignore
                param_type=param_type,  # type: ignore
                param_description=doc_str_return.description,  # type: ignore
            )
    if len(parsed_doc_str.raises):
        doc_str_frag += "\n### Raises"
        for raises in parsed_doc_str.raises:
            doc_str_frag += gen_param_set(
                param_name="",
                param_type=[raises.type_name],  # type: ignore
                param_description=raises.description,  # type: ignore
            )
    if parsed_doc_str.deprecation is not None:
        raise NotImplementedError("Deprecation not implemented.")
    metas: list[docstring_parser.common.DocstringMeta] = []
    for met in parsed_doc_str.meta:
        if not isinstance(
            met,
            (
                docstring_parser.common.DocstringParam,
                docstring_parser.common.DocstringDeprecated,
                docstring_parser.common.DocstringRaises,
                docstring_parser.common.DocstringReturns,
            ),
        ):
            metas.append(met)
    if metas:
        doc_str_frag += "\n### Notes"
        for meta in metas:
            doc_str_frag += strip_markdown(meta.description)  # type: ignore

    return doc_str_frag


def custom_format_signature(sig: inspect.Signature, colon: bool = True) -> str:
    """pdoc currently returns expanded annotations - problematic for npt.Arraylike etc."""
    # First get a list with all params as strings.
    params: list[str] = doc._PrettySignature._params(sig)  # type: ignore
    return_annot = doc._PrettySignature._return_annotation_str(sig)  # type: ignore
    parsed_return_annot: list[str] = []
    if return_annot not in ["", "None", None]:
        ra = return_annot.lstrip("tuple[")
        ra = ra.rstrip("]")
        rs = ra.split(",")
        for r in rs:
            r = r.strip()
            if "." in r:
                r = r.split(".")[-1]
            if r.lower() not in ["any", "nonetype"]:
                parsed_return_annot.append(r)
    # build tags
    if len(params) <= 1 and len(parsed_return_annot) <= 1:
        sig_fragment: tags.div = tags.div(cls="signature")
    else:
        sig_fragment: tags.div = tags.div(cls="signature multiline")
    with sig_fragment:
        tags.span("(", cls="pt")
    # nest sig params for CSS alignment
    for param in params:
        param_fragment = tags.div(cls="param")
        if ":" in param:
            param_text, annot = param.split(":")
            if "any" in annot.strip().lower():
                annot = None
            elif annot.strip().lower().startswith("union"):
                annot = None
        else:
            param_text = param
            annot = None
        with param_fragment:
            tags.span(param_text, cls="pn")
        if annot is not None:
            with param_fragment:
                tags.span(":", cls="pc")
                tags.span(annot, cls="pa")
        sig_fragment += param_fragment
    if not parsed_return_annot:
        with sig_fragment:
            tags.span(")", cls="pt")
    else:
        with sig_fragment:
            tags.span(")->[", cls="pt")
            for parsed_return in parsed_return_annot:
                tags.span(parsed_return, cls="pr")
            tags.span("]", cls="pt")

    return sig_fragment.render()


if __name__ == "__main__":
    # Add custom function
    render.env.filters["custom_process_docstring"] = custom_process_docstring  # type: ignore
    render.env.filters["custom_format_signature"] = custom_format_signature  # type: ignore
    here = Path(__file__).parent

    module_file_maps = [
        ("cityseer.rustalgos", here / "src/pages/rustalgos/rustalgos.md"),
        ("cityseer.metrics.observe", here / "src/pages/metrics/observe.md"),
        ("cityseer.metrics.networks", here / "src/pages/metrics/networks.md"),
        ("cityseer.metrics.layers", here / "src/pages/metrics/layers.md"),
        ("cityseer.metrics.visibility", here / "src/pages/metrics/visibility.md"),
        ("cityseer.tools.graphs", here / "src/pages/tools/graphs.md"),
        ("cityseer.tools.io", here / "src/pages/tools/io.md"),
        ("cityseer.tools.plot", here / "src/pages/tools/plot.md"),
        ("cityseer.tools.mock", here / "src/pages/tools/mock.md"),
        ("cityseer.tools.util", here / "src/pages/tools/util.md"),
    ]
    for module_name, output_path in module_file_maps:
        render.configure(template_directory=here / "pdoc_templates", docformat="numpy", math=True)

        module = importlib.import_module(module_name)
        d = doc.Module(module)
        out = render.html_module(module=d, all_modules={module_name: d})
        with open(output_path, "w") as f:
            f.write(out)
