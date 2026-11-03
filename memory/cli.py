import argparse
import os
from typing import List

from memory.adapters.memory_op_seeds import seed_memory_ops
from memory.adapters.migrate_faiss import migrate_faiss
from memory.service import MemoryService

DEFAULT_DB = None  # MemoryService puts it at <project-dir>/.jarvis/memory.sqlite


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=DEFAULT_DB,
                        help="memory database path (default <project-dir>/.jarvis/memory.sqlite)")
    common.add_argument("--project-dir", default=os.getcwd(), help="project directory")
    parser = argparse.ArgumentParser(prog="python -m memory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report", parents=[common], help="print memory statistics")
    sub.add_parser("migrate", parents=[common], help="import FAISS route examples, seed ops")
    sleep = sub.add_parser("sleep", parents=[common], help="run a consolidation pass")
    sleep.add_argument("--dry-run", action="store_true", help="run on a copy, write nothing")
    sub.add_parser("reset-params", parents=[common], help="restore default params")
    return parser


def _migrate(args) -> None:
    from adapters.embeddings import HuggingFaceAdapter  # needs semantic_router; lazy on purpose

    embedder = HuggingFaceAdapter()
    service = MemoryService(None, args.project_dir, args.db, start_threads=False)
    try:
        metadata = os.path.join(args.project_dir, ".vectorstore", "metadata.json")
        routes = migrate_faiss(service.store, embedder, metadata)
        ops = seed_memory_ops(service.store, embedder, service.clock)
        print("Imported %d route examples, seeded %d memory-op examples." % (routes, ops))
    finally:
        service.close()


def _run(args) -> None:
    service = MemoryService(None, args.project_dir, args.db, start_threads=False)
    try:
        if args.command == "report":
            print(service.report(), end="")
        elif args.command == "sleep":
            r = service.sleep.run(dry_run=args.dry_run)
            print("Sleep%s: rescored %d, summarised %d, deduped %d, purged %d, tune %s" % (
                " (dry run)" if r.dry_run else "", r.rescored, r.summarised, r.deduped,
                r.purged, r.tune.get("status")))
        else:
            service.reset_params()
            print("Params reset to defaults.")
    finally:
        service.close()


def main(argv: List[str]) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "migrate":
            _migrate(args)
        else:
            _run(args)
    except Exception as e:
        print("error: %s" % e)
        return 1
    return 0
