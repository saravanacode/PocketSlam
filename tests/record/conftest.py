from collections import Counter
from pathlib import Path

from mcap.reader import make_reader


def read_topics(path: Path) -> Counter[str]:
    with path.open("rb") as f:
        return Counter(channel.topic for _, channel, _ in make_reader(f).iter_messages())
