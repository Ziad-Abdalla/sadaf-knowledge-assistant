"""A small Arabic (right-to-left) chat page with a role switcher.

The switcher stands in for the company login: in production the user and roles come from
SSO / Odoo groups, and nobody can pick their own role.
"""

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import config
from .access import load_users
from .assistant import Assistant

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

EXAMPLES = [
    "كم المياومة لو رحت أصور في أبها؟",
    "متى لازم أرجّع الكاميرا بعد نهاية التصوير؟",
    "كم ميزانية مسلسل رمال الذاكرة؟",
    "What loudness level does the TV channel require for delivery?",
    "كم كانت المياومة داخل المملكة في 2025 وكم صارت الآن؟",
    "هل يوجد موقف سيارات للموظفين؟",
]

REASONS = {
    "no_relevant_source": "لا يوجد في المستندات المتاحة لك نص قريب بما يكفي من سؤالك "
                          "(أعلى تشابه {best:.2f} وهو أقل من الحد {min:.2f}). لم يُستدعَ النموذج أصلًا.",
    "model_not_in_documents": "وجدت نصوصًا قريبة من سؤالك، لكنها لا تحتوي على الإجابة نفسها، "
                              "فاخترت ألا أخمّن.",
}


def create_app(assistant: Assistant | None = None, users: dict | None = None) -> FastAPI:
    app = FastAPI(title="Sadaf knowledge assistant (demo)")
    state = {"assistant": assistant}
    users = users or load_users(config.USERS_FILE)

    def get_assistant() -> Assistant:
        if state["assistant"] is None:
            state["assistant"] = Assistant.from_config()
        return state["assistant"]

    def page(request: Request, **ctx):
        a = get_assistant()
        return TEMPLATES.TemplateResponse(request, "chat.html", {
            "users": users, "examples": EXAMPLES, "release": a.release, **ctx})

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, user_id: str = "saad"):
        return page(request, user_id=user_id if user_id in users else "saad", answer=None,
                    question="", history=False)

    @app.post("/ask", response_class=HTMLResponse)
    def ask(request: Request, question: str = Form(...), user_id: str = Form("saad"),
            history: str | None = Form(None)):
        return answer_page(request, question, user_id, history)

    @app.get("/ask", response_class=HTMLResponse)
    def ask_link(request: Request, q: str, user_id: str = "saad", history: str | None = None):
        """Same as the form, as a link (handy for sharing a question, and for screenshots)."""
        return answer_page(request, q, user_id, history)

    def answer_page(request: Request, question: str, user_id: str, history: str | None):
        user = users.get(user_id) or users["saad"]
        a = get_assistant()
        answer = a.ask(question, user, include_history=True if history else None)
        why = REASONS.get(answer.reason, "").format(best=answer.best_dense, min=a.release.min_dense)
        return page(request, user_id=user.user_id, answer=answer, question=question,
                    history=bool(history), why=why)

    @app.get("/admin/audit", response_class=HTMLResponse)
    def audit(request: Request):
        """Demo only. In production this page sits behind an admin role."""
        a = get_assistant()
        return TEMPLATES.TemplateResponse(request, "audit.html", {
            "rows": a.audit.recent(30), "flags": a.audit.open_flags(),
            "summary": a.audit.summary_by("release_id"), "release": a.release})

    return app
