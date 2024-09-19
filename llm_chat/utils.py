import asyncio
import uuid
from typing import (AsyncGenerator, Awaitable, Callable, Optional, Tuple, TypeVar)
from typing import Sequence as GenericSequence
from asyncio import FIRST_COMPLETED, ensure_future
from dataclasses import dataclass

T = TypeVar("T")

# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

async def merge_async_iterators(
    *iterators: AsyncGenerator[T, None],
    is_cancelled: Optional[Callable[[], Awaitable[bool]]] = None,
) -> AsyncGenerator[Tuple[int, T], None]:
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.

    It also optionally polls a provided function at least once per second
    to check for client cancellation.
    """

    # Can use anext() in python >= 3.10
    awaits = {
        ensure_future(pair[1].__anext__()): pair
        for pair in enumerate(iterators)
    }
    timeout = None if is_cancelled is None else 1
    try:
        while awaits:
            done, pending = await asyncio.wait(awaits.keys(),
                                               return_when=FIRST_COMPLETED,
                                               timeout=timeout)
            if is_cancelled is not None and await is_cancelled():
                raise asyncio.CancelledError("client cancelled")
            for d in done:
                pair = awaits.pop(d)
                try:
                    item = await d
                    i, it = pair
                    awaits[ensure_future(it.__anext__())] = pair
                    yield i, item
                except StopAsyncIteration:
                    pass
    finally:
        # Cancel any remaining iterators
        for f, (_, it) in awaits.items():
            with contextlib.suppress(BaseException):
                f.cancel()
                await it.aclose()