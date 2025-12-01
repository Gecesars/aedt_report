from __future__ import annotations

from typing import Iterable, Sequence

from flask import current_app, render_template
from flask_mailman import EmailMessage


def _send_message(message: EmailMessage, category: str, recipients: Sequence[str]) -> bool:
    logger = current_app.logger
    try:
        message.send()
        logger.info("Email %s enviado para %s", category, ", ".join(recipients))
        return True
    except Exception as exc:  # pragma: no cover - log for observability
        logger.exception("Falha ao enviar email %s para %s: %s", category, ", ".join(recipients), exc)
        return False


def send_confirmation_email(email: str, token: str) -> None:
    subject = "Confirme seu e-mail EFTX"
    confirm_url = current_app.url_for("auth.confirm_email", token=token, _external=True)
    html_body = render_template("emails/confirm_email.html", confirm_url=confirm_url)
    message = EmailMessage(subject=subject, body=html_body, to=[email])
    message.content_subtype = "html"
    _send_message(message, "confirmacao", [email])


def send_admin_cnpj_alert(email: str, user_name: str) -> None:
    admins = current_app.config.get("ADMIN_ALERT_RECIPIENTS")
    if not admins:
        current_app.logger.info("Sem destinatarios para alerta de CNPJ; ignorando aviso para %s", email)
        return
    if isinstance(admins, str):
        admins = [addr.strip() for addr in admins.split(",") if addr.strip()]
    if not admins:
        current_app.logger.info("Sem destinatarios validos para alerta de CNPJ apos parser; ignorando %s", email)
        return
    subject = "Novo usuario aguardando validacao de CNPJ"
    html_body = render_template("emails/admin_cnpj_alert.html", email=email, user_name=user_name)
    message = EmailMessage(subject=subject, body=html_body, to=list(admins))
    message.content_subtype = "html"
    _send_message(message, "alerta_cnpj", list(admins))
