from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")

def tqdm(iterable: Iterable[T], *args: Any, **kwargs: Any) -> Iterator[T]: ...
