import threading
from pathlib import Path

from optmod.schemas import LogEntry


class RoutingLog:
    def __init__(self, path: str = "routing.log.jsonl") -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def append(self, entry: LogEntry) -> None:
        line = entry.to_jsonl() + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)

    def clear(self) -> None:
        with self._lock:
            self._path.write_text("")

    def flush(self) -> None:
        pass
