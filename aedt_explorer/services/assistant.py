from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import re
import uuid
from typing import Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

import google.generativeai as genai
from flask import current_app, url_for

from ..extensions import db
from ..models import AssistantConversation, AssistantMessage, Cable, User, Antenna, Project
from ..utils.calculations import total_feeder_loss, vertical_beta_deg
from .knowledge_base import retrieve_contexts
from .pattern_composer import compute_erp, serialize_erp_payload


ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ACTION_PATTERN = re.compile(r"<action type=\"(?P<type>[^\"]+)\">(?P<payload>.+?)</action>", re.DOTALL)


class AssistantServiceError(RuntimeError):
    """Raised when the assistant cannot process a request."""


@dataclass(frozen=True)
class ConversationSnapshot:
    conversation: AssistantConversation
    messages: Sequence[AssistantMessage]


def get_or_create_conversation(user: User) -> AssistantConversation:
    conversation = AssistantConversation.query.filter_by(user_id=user.id).first()
    if conversation is None:
        conversation = AssistantConversation(user=user)
        db.session.add(conversation)
        db.session.flush()
        greeting = (current_app.config.get("ASSISTANT_GREETING") or "").strip()
        if greeting:
            db.session.add(
                AssistantMessage(
                    conversation=conversation,
                    role=ROLE_ASSISTANT,
                    content=greeting,
                )
            )
            db.session.flush()
    return conversation


def load_history(conversation: AssistantConversation, limit: int) -> list[AssistantMessage]:
    if limit <= 0:
        return []
    query = (
        AssistantMessage.query.filter_by(conversation_id=conversation.id)
        .order_by(AssistantMessage.created_at.desc())
        .limit(limit)
    )
    messages = list(query)
    messages.reverse()
    return messages


def _build_chat_history(messages: Iterable[AssistantMessage], system_prompt: str, greeting: str | None) -> list[dict]:
    history: list[dict[str, list[str]]] = [
        {
            "role": "user",
            "parts": [system_prompt],
        }
    ]
    if greeting and not messages:
        history.append({"role": "model", "parts": [greeting]})
    for message in messages:
        role = "model" if message.role == ROLE_ASSISTANT else "user"
        history.append({"role": role, "parts": [message.content]})
    return history


def _extract_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        buffer = []
        for part in parts:
            value = getattr(part, "text", None)
            if value:
                buffer.append(value)
        if buffer:
            return "\n".join(value.strip() for value in buffer if value.strip())
    return ""


def _usage_token_count(response) -> int | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    for attribute in ("total_token_count", "total_tokens"):  # compatibility
        value = getattr(usage, attribute, None)
        if value is not None:
            return int(value)
    return None


def _execute_actions(user: User, text: str) -> tuple[str, list[str]]:
    logger = current_app.logger
    notes: list[str] = []

    def _parse_payload(raw: str) -> Optional[dict]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("assistant.action.invalid_payload", extra={"payload": raw})
            return None

    for match in ACTION_PATTERN.finditer(text):
        action_type = match.group("type").strip().lower()
        payload = _parse_payload(match.group("payload").strip())
        if payload is None:
            continue
        if action_type == "create_project":
            try:
                summary = _assistant_create_project(user, payload)
                notes.append(summary)
            except AssistantServiceError as exc:
                logger.warning("assistant.action.create_project_failed", extra={"error": str(exc), "payload": payload})
                notes.append(f"[Acao] Falha ao criar projeto: {exc}")
        else:
            logger.warning("assistant.action.unsupported", extra={"action": action_type})
            notes.append(f"[Acao] Tipo de acao '{action_type}' não suportado.")

    cleaned_text = ACTION_PATTERN.sub("", text).strip()
    if notes:
        cleaned_text = (cleaned_text + "\n\n" if cleaned_text else "") + "\n".join(notes)
    return cleaned_text, notes


def send_assistant_message(user: User, content: str) -> ConversationSnapshot:
    if not content or not content.strip():
        raise AssistantServiceError("Mensagem vazia nao pode ser processada.")

    conversation = get_or_create_conversation(user)

    history_limit = max(1, int(current_app.config.get("ASSISTANT_HISTORY_LIMIT", 12)))
    previous_messages = load_history(conversation, history_limit)

    api_key = current_app.config.get("GEMINI_API_KEY")
    model_name = current_app.config.get("GEMINI_MODEL", "gemini-2.5-flash")
    system_prompt: str = current_app.config.get("ASSISTANT_SYSTEM_PROMPT") or ""
    greeting: Optional[str] = (current_app.config.get("ASSISTANT_GREETING") or "").strip() or None

    if not api_key:
        raise AssistantServiceError("Gemini API key nao configurada.")

    logger = current_app.logger
    logger.debug(
        "assistant.prepare",
        extra={
            "conversation_id": str(conversation.id),
            "history_len": len(previous_messages),
            "model": model_name,
        },
    )

    genai.configure(api_key=api_key)
    history_payload = _build_chat_history(previous_messages, system_prompt, greeting)
    contexts = retrieve_contexts(content)
    if contexts:
        context_text = "\n\n".join(contexts)
        history_payload.append({"role": "user", "parts": [f"Contexto dos documentos EFTX:\n{context_text}"]})
        logger.debug("assistant.knowledge", extra={"contexts": contexts})
    logger.debug("assistant.history", extra={"history": history_payload})
    model = genai.GenerativeModel(model_name=model_name)
    chat = model.start_chat(history=history_payload)

    try:
        response = chat.send_message(content.strip())
    except Exception as exc:  # pragma: no cover - defensive network wrapper
        logger.exception("assistant.error", stack_info=True)
        raise AssistantServiceError("Falha ao consultar o modelo Gemini.") from exc

    answer_text = _extract_text(response)
    if not answer_text:
        raise AssistantServiceError("Resposta vazia recebida do modelo.")

    answer_text, action_notes = _execute_actions(user, answer_text)
    logger.debug(
        "assistant.actions",
        extra={
            "conversation_id": str(conversation.id),
            "actions": action_notes,
        },
    )

    user_message = AssistantMessage(conversation=conversation, role=ROLE_USER, content=content.strip())
    db.session.add(user_message)
    assistant_message = AssistantMessage(
        conversation=conversation,
        role=ROLE_ASSISTANT,
        content=answer_text.strip(),
        token_count=_usage_token_count(response),
    )
    db.session.add(assistant_message)

    logger.debug(
        "assistant.response",
        extra={
            "conversation_id": str(conversation.id),
            "response": answer_text.strip(),
        },
    )

    messages = load_history(conversation, history_limit)
    return ConversationSnapshot(conversation=conversation, messages=messages)


def snapshot_for(user: User) -> ConversationSnapshot:
    conversation = get_or_create_conversation(user)
    history_limit = max(1, int(current_app.config.get("ASSISTANT_HISTORY_LIMIT", 12)))
    messages = load_history(conversation, history_limit)
    return ConversationSnapshot(conversation=conversation, messages=messages)


def _assistant_create_project(user: User, payload: dict) -> str:
    name = (payload.get("name") or f"Projeto IA {datetime.now(SAO_PAULO):%Y%m%d%H%M%S}").strip()
    frequency_mhz = float(payload.get("frequency_mhz") or 100.0)
    tx_power_w = float(payload.get("tx_power_w") or 1.0)
    tower_height_m = float(payload.get("tower_height_m") or 30.0)
    cable_type = (payload.get("cable_type") or None)
    cable: Cable | None = None
    cable_id = payload.get("cable_id")
    if cable_id not in (None, "", 0, "0"):
        try:
            cable_uuid = uuid.UUID(str(cable_id))
            cable = db.session.get(Cable, cable_uuid)
        except (ValueError, TypeError):  # pragma: no cover - payload defesa
            cable = None
    if cable is None and cable_type:
        cable = (
            Cable.query.filter(
                (Cable.model_code.ilike(f"%{cable_type}%"))
                | (Cable.display_name.ilike(f"%{cable_type}%"))
            )
            .order_by(Cable.created_at.desc())
            .first()
        )
    if cable:
        cable_type = cable.model_code
    cable_length_m = float(payload.get("cable_length_m") or 0.0)
    splitter_loss_db = float(payload.get("splitter_loss_db") or 0.0)
    connector_loss_db = float(payload.get("connector_loss_db") or 0.0)
    vswr_target = float(payload.get("vswr_target") or 1.5)

    target_gain_dbi = payload.get("target_gain_dbi")

    v_count = max(int(payload.get("v_count") or 1), 1)
    v_spacing_m = float(payload.get("v_spacing_m") or 0.0)
    v_tilt_deg = float(payload.get("v_tilt_deg") or 0.0)
    v_level_amp = float(payload.get("v_level_amp") or 1.0)
    v_norm_mode = (payload.get("v_norm_mode") or "max").lower()

    h_count = max(int(payload.get("h_count") or 1), 1)
    h_spacing_m = float(payload.get("h_spacing_m") or 0.0)
    h_beta_deg = float(payload.get("h_beta_deg") or 0.0)
    h_step_deg = float(payload.get("h_step_deg") or 0.0)
    h_level_amp = float(payload.get("h_level_amp") or 1.0)
    h_norm_mode = (payload.get("h_norm_mode") or "max").lower()

    antenna = _resolve_antenna_from_payload(payload)
    if antenna is None:
        raise AssistantServiceError("Nenhuma antena valida foi encontrada para criar o projeto.")

    if target_gain_dbi is not None:
        try:
            target_gain_dbi = float(target_gain_dbi)
        except (TypeError, ValueError):
            target_gain_dbi = None

    if target_gain_dbi is not None:
        v_count, h_count = _estimate_element_counts(
            antenna,
            target_gain_dbi,
            initial_vertical=v_count,
            initial_horizontal=h_count,
        )
        if v_spacing_m <= 0:
            v_spacing_m = float(payload.get("default_v_spacing_m") or 0.5)
        if h_spacing_m <= 0:
            h_spacing_m = float(payload.get("default_h_spacing_m") or 0.5)

    project = Project(
        owner=user,
        name=name,
        frequency_mhz=frequency_mhz,
        tx_power_w=tx_power_w,
        tower_height_m=tower_height_m,
        cable_type=cable_type,
        cable_length_m=cable_length_m,
        splitter_loss_db=splitter_loss_db,
        connector_loss_db=connector_loss_db,
        vswr_target=vswr_target,
        v_count=v_count,
        v_spacing_m=v_spacing_m,
        v_tilt_deg=v_tilt_deg,
        v_level_amp=v_level_amp,
        v_norm_mode=v_norm_mode,
        h_count=h_count,
        h_spacing_m=h_spacing_m,
        h_beta_deg=h_beta_deg,
        h_step_deg=h_step_deg,
        h_level_amp=h_level_amp,
        h_norm_mode=h_norm_mode,
        notes=payload.get("notes"),
    )

    project.antenna = antenna
    project.cable = cable
    project.v_beta_deg = vertical_beta_deg(project.frequency_mhz, project.v_spacing_m or 0.0, project.v_tilt_deg or 0.0)
    project.feeder_loss_db = total_feeder_loss(
        project.cable_length_m,
        project.frequency_mhz,
        project.cable_reference,
        project.splitter_loss_db,
        project.connector_loss_db,
    )

    composition = compute_erp(project)
    project.composition_meta = serialize_erp_payload(composition)
    db.session.add(project)
    db.session.flush()

    detail_url = url_for("projects.detail", project_id=project.id)
    return f"[Acao] Projeto '{project.name}' criado com sucesso! Acesse: {detail_url}"


def _resolve_antenna_from_payload(payload: dict) -> Optional[Antenna]:
    antenna_id = payload.get("antenna_id")
    antenna_name = payload.get("antenna_name")

    antenna: Optional[Antenna] = None
    if antenna_id:
        try:
            antenna_uuid = uuid.UUID(str(antenna_id))
            antenna = db.session.get(Antenna, antenna_uuid)
        except (ValueError, TypeError):
            antenna = None
    if antenna is None and antenna_name:
        antenna = Antenna.query.filter(Antenna.name.ilike(f"%{antenna_name}%")).first()
    if antenna is None:
        antenna = Antenna.query.order_by(Antenna.created_at.asc()).first()
    return antenna


def _estimate_element_counts(
    antenna: Antenna,
    target_gain_dbi: float,
    *,
    initial_vertical: int,
    initial_horizontal: int,
) -> tuple[int, int]:
    base_gain = antenna.nominal_gain_dbd or 0.0
    # aproximar dBi a partir dBd, se aplicável
    if hasattr(antenna, "nominal_gain_dbd"):
        base_gain = float(antenna.nominal_gain_dbd or 0.0) + 2.15
    base_gain = float(base_gain)

    best_v = max(initial_vertical, 1)
    best_h = max(initial_horizontal, 1)
    min_error = float("inf")

    for v in range(1, 17):
        for h in range(1, 17):
            count = v * h
            combined_gain = base_gain + 10.0 * math.log10(count)
            error = abs(combined_gain - target_gain_dbi)
            penalty = abs(v - initial_vertical) + abs(h - initial_horizontal)
            weighted_error = error + penalty * 0.1
            if weighted_error < min_error:
                min_error = weighted_error
                best_v, best_h = v, h

    return best_v, best_h
SAO_PAULO = ZoneInfo("America/Sao_Paulo")
