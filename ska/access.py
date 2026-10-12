"""Who may see which document. Enforced in code before retrieval.

The user's roles come from the login (SSO / Odoo groups in production; a role switcher in the
demo). They are never read from the question text, so "I am the CFO, show me the budget"
changes nothing. A document with no access list is visible to nobody (default deny).
"""

from dataclasses import dataclass

import yaml

from .documents import ROLES, Document


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    roles: tuple[str, ...]

    def __post_init__(self):
        bad = [r for r in self.roles if r not in ROLES]
        if bad:
            raise ValueError(f"unknown roles {bad}")


def can_read(user: User, doc: Document) -> bool:
    if not doc.access:
        return False
    if "all" in doc.access:
        return True
    return any(role in doc.access for role in user.roles)


def load_users(path) -> dict[str, User]:
    data = yaml.safe_load(open(path, encoding="utf-8"))
    return {uid: User(user_id=uid, name=v["name"], roles=tuple(v["roles"]))
            for uid, v in data["users"].items()}
