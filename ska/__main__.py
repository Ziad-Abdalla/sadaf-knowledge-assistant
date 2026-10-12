"""Command line.

  python -m ska ingest [--embed-model NAME]      build the index for the active release
  python -m ska ask "question" --user saad [--history]
  python -m ska eval [--retrieval-only] [--no-judge] [--embed-model NAME]
  python -m ska audit [--last 10 | --summary release_id|roles | --flags]
  python -m ska release list | use <id>            (use = rollback / roll forward)
  python -m ska serve [--port 8000]
"""

import argparse
import json
import sys

from . import config


def _users():
    from .access import load_users
    return load_users(config.USERS_FILE)


def cmd_ingest(args):
    from .embeddings import load_embedder
    from .index import build_index, index_path

    name = args.embed_model or config.active_release().embed_model
    info = build_index(config.CORPUS_DIR, load_embedder(name), index_path(config.INDEX_DIR, name))
    print(f"Indexed {info['docs']} documents into {info['chunks']} chunks -> {info['path']}")


def cmd_ask(args):
    from .assistant import Assistant

    a = Assistant.from_config()
    ans = a.ask(args.question, _users()[args.user], include_history=True if args.history else None)
    print(ans.text)
    print(f"\n[{ans.status}{' / ' + ans.reason if ans.reason else ''}] release={ans.release_id} "
          f"model={ans.model} prompt={ans.prompt_hash} best_dense={ans.best_dense:.3f} "
          f"cost=${ans.cost_usd:.5f} {ans.latency_ms} ms")


def cmd_eval(args):
    from .embeddings import load_embedder
    from .evaluation import load_items, run_full, run_retrieval
    from .index import Index, index_path

    release = config.active_release()
    embed_name = args.embed_model or release.embed_model
    if args.embed_model:
        release = config.Release(**{**release.__dict__, "embed_model": embed_name})
    embedder = load_embedder(embed_name)
    index = Index.load(index_path(config.INDEX_DIR, embed_name))
    items, users = load_items(), _users()
    r = run_retrieval(index, embedder, release, items, users)
    print(json.dumps(r["metrics"], ensure_ascii=False, indent=1))
    if args.verbose:
        for row in r["rows"]:
            print(row["id"], row["type"], row["best_dense"], row["first_gold_rank"], row["ranked"][:3])
    if args.retrieval_only:
        return
    from .assistant import Assistant
    from .audit import AuditLog

    a = Assistant(index, embedder, release, AuditLog(config.AUDIT_DB))
    out = run_full(a, items, users, with_judge=not args.no_judge, retrieval_metrics=r["metrics"])
    m = out["metrics"]
    print(json.dumps(m, ensure_ascii=False, indent=1))
    sys.exit(0 if m["gate"] == "pass" else 1)


def cmd_audit(args):
    from .audit import AuditLog

    log = AuditLog(config.AUDIT_DB)
    if args.flags:
        rows = log.open_flags()
    elif args.summary:
        rows = log.summary_by(args.summary)
    else:
        rows = log.recent(args.last)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


def cmd_release(args):
    active, releases = config.load_releases()
    if args.action == "use":
        config.set_active_release(args.id)
        print(f"Active release is now {args.id} (was {active}).")
        return
    for rid, r in releases.items():
        mark = "*" if rid == active else " "
        print(f"{mark} {rid}  model={r.llm_model} prompt={r.prompt}@{r.prompt_hash} "
              f"embed={r.embed_model} min_dense={r.min_dense}  {r.notes}")


def cmd_serve(args):
    import uvicorn
    from .web import create_app

    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ska")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("--embed-model"); s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("--user", default="saad")
    s.add_argument("--history", action="store_true"); s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("eval"); s.add_argument("--retrieval-only", action="store_true")
    s.add_argument("--no-judge", action="store_true"); s.add_argument("--embed-model")
    s.add_argument("-v", "--verbose", action="store_true"); s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("audit"); s.add_argument("--last", type=int, default=10)
    s.add_argument("--summary"); s.add_argument("--flags", action="store_true"); s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("release"); s.add_argument("action", choices=["list", "use"])
    s.add_argument("id", nargs="?"); s.set_defaults(fn=cmd_release)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000); s.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
