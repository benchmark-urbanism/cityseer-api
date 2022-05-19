from typing import Any, Iterable, Iterator, TypeVar

_T = TypeVar("_T")

def tqdm(iterable: Iterable[_T], *args: Any, **kwargs: Any) -> Iterator[_T]: ...
