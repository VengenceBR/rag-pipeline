from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class DebouncedIngestHandler(FileSystemEventHandler):
    """Collapses a burst of filesystem events (an editor's save-then-rename, a bulk
    file copy) into a single ingest call, fired once no new event has arrived for
    `debounce_seconds`. Independent of any real Observer/filesystem, so it's
    unit-testable by calling on_any_event() directly with a synthetic event.
    """

    def __init__(self, callback: Callable[[], None], debounce_seconds: float = 3.0):
        self._callback = callback
        self._debounce_seconds = debounce_seconds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        try:
            self._callback()
        except Exception:
            logger.exception("Ingest callback failed; will retry on the next change.")

    def on_any_event(self, event) -> None:
        if event.is_directory:
            return
        self._schedule()

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


def _ingest_callback(service, directory: Path, company_id: str) -> Callable[[], None]:
    def do_ingest() -> None:
        logger.info("Change detected under %s - re-ingesting company '%s'", directory, company_id)
        result = service.ingest(directory, company_id, sync=True)
        logger.info("Re-ingest for '%s' done: %s", company_id, result.model_dump_json())

    return do_ingest


def watch_companies(company_dirs: dict[str, Path], debounce_seconds: float = 3.0) -> None:
    """Blocks forever. Watches each company_id -> directory pair, re-ingesting in
    sync mode (so deleted/edited files are reflected, not just additions) after
    debounce_seconds of quiet following the last change under that directory.
    Runs an initial sync for every company immediately, so the index matches disk
    state on startup rather than waiting for the first future change.
    """
    from rag.service import RAGService

    if not company_dirs:
        raise ValueError("No company directories to watch.")

    service = RAGService()
    observer = Observer()
    handlers = []

    for company_id, directory in company_dirs.items():
        do_ingest = _ingest_callback(service, directory, company_id)
        do_ingest()  # initial sync so startup state matches disk immediately

        handler = DebouncedIngestHandler(do_ingest, debounce_seconds=debounce_seconds)
        handlers.append(handler)
        observer.schedule(handler, str(directory), recursive=True)
        logger.info("Watching %s for company '%s'", directory, company_id)

    observer.start()
    try:
        while observer.is_alive():
            observer.join(timeout=1.0)
    finally:
        for handler in handlers:
            handler.stop()
        observer.stop()
        observer.join()


def discover_company_dirs(data_dir: Path) -> dict[str, Path]:
    """Maps each immediate subdirectory of data_dir to a company_id (the folder
    name) e.g. data/acme/ -> company 'acme'.
    """
    return {p.name: p for p in sorted(data_dir.iterdir()) if p.is_dir()}
