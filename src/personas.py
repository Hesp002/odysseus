"""Chat personas — selectable characters with their own memory.

A persona is a ``CrewMember`` row (the table was designed for "a custom AI
persona with its own personality, model, tools, and memory scope"); the
per-owner personal assistant (``is_default_assistant``) is excluded. A chat is
bound to a persona when it is created via ``sessions.crew_member_id``.

"Odysseus" is the default persona: no crew member, no persona prompt, and the
memories that carry no ``persona`` key. A persona's memories carry
``persona=<crew_member.id>`` and are only visible in that persona's chats.
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterable, List, Optional

from core.database import CrewMember, SessionLocal
from core.database import Session as DbSession

logger = logging.getLogger(__name__)

DEFAULT_PERSONA_NAME = "Odysseus"


def persona_to_dict(crew: CrewMember) -> dict:
    return {
        "id": crew.id,
        "name": crew.name,
        "personality": crew.personality or "",
        "avatar": crew.avatar,
        "sort_order": crew.sort_order or 0,
    }


def _persona_query(db, owner: Optional[str]):
    q = db.query(CrewMember).filter(
        (CrewMember.is_default_assistant == False)  # noqa: E712
        | (CrewMember.is_default_assistant.is_(None))
    )
    if owner:
        q = q.filter(CrewMember.owner == owner)
    return q


def list_personas(owner: Optional[str]) -> List[dict]:
    db = SessionLocal()
    try:
        rows = _persona_query(db, owner).order_by(
            CrewMember.sort_order, CrewMember.name
        ).all()
        return [persona_to_dict(r) for r in rows]
    finally:
        db.close()


def get_persona(persona_id: Optional[str], owner: Optional[str] = None) -> Optional[dict]:
    if not persona_id:
        return None
    db = SessionLocal()
    try:
        row = _persona_query(db, owner).filter(CrewMember.id == persona_id).first()
        return persona_to_dict(row) if row else None
    finally:
        db.close()


def create_persona(owner: Optional[str], name: str, personality: str = "") -> dict:
    db = SessionLocal()
    try:
        crew = CrewMember(
            id=f"persona-{uuid.uuid4().hex[:12]}",
            owner=owner,
            name=name,
            personality=personality,
            is_default_assistant=False,
            is_active=True,
        )
        db.add(crew)
        db.commit()
        db.refresh(crew)
        return persona_to_dict(crew)
    finally:
        db.close()


def update_persona(
    persona_id: str,
    owner: Optional[str],
    *,
    name: Optional[str] = None,
    personality: Optional[str] = None,
) -> Optional[dict]:
    db = SessionLocal()
    try:
        crew = _persona_query(db, owner).filter(CrewMember.id == persona_id).first()
        if not crew:
            return None
        if name is not None:
            crew.name = name
        if personality is not None:
            crew.personality = personality
        db.commit()
        db.refresh(crew)
        return persona_to_dict(crew)
    finally:
        db.close()


def delete_persona(persona_id: str, owner: Optional[str]) -> bool:
    """Delete the persona row and unlink its chats (they fall back to Odysseus).

    The caller removes the persona's memories (see ``remove_persona_memories``).
    """
    db = SessionLocal()
    try:
        crew = _persona_query(db, owner).filter(CrewMember.id == persona_id).first()
        if not crew:
            return False
        db.query(DbSession).filter(DbSession.crew_member_id == persona_id).update(
            {DbSession.crew_member_id: None}, synchronize_session=False
        )
        db.delete(crew)
        db.commit()
        return True
    finally:
        db.close()


def session_persona(session_id: Optional[str]) -> Optional[dict]:
    """The persona a chat was created with, or None for Odysseus."""
    if not session_id:
        return None
    db = SessionLocal()
    try:
        row = db.query(DbSession.crew_member_id, DbSession.owner).filter(
            DbSession.id == session_id
        ).first()
        if not row or not row[0]:
            return None
        crew = _persona_query(db, row[1]).filter(CrewMember.id == row[0]).first()
        return persona_to_dict(crew) if crew else None
    except Exception as e:
        logger.debug("session persona lookup failed for %s: %s", session_id, e)
        return None
    finally:
        db.close()


def session_persona_id(session_id: Optional[str]) -> Optional[str]:
    persona = session_persona(session_id)
    return persona["id"] if persona else None


def set_session_persona(session_id: str, persona_id: Optional[str]) -> None:
    db = SessionLocal()
    try:
        db.query(DbSession).filter(DbSession.id == session_id).update(
            {DbSession.crew_member_id: persona_id or None}, synchronize_session=False
        )
        db.commit()
    finally:
        db.close()


def persona_system_prompt(persona: Optional[dict]) -> Optional[str]:
    if not persona:
        return None
    name = (persona.get("name") or "").strip()
    personality = (persona.get("personality") or "").strip()
    parts = []
    if name:
        parts.append(f"Your name is {name}.")
    if personality:
        parts.append(personality)
    parts.append(
        "Stay in character for the whole conversation. You are talking with "
        "the user directly; the saved memories you are given are what you, "
        f"{name or 'this persona'}, remember about them."
    )
    return " ".join(parts)


# ── Memory scoping ──────────────────────────────────────────────────

def memory_persona(entry: dict) -> Optional[str]:
    """Persona id a memory belongs to; None means Odysseus (the default)."""
    return entry.get("persona") or None


def filter_for_persona(entries: Iterable[dict], persona_id: Optional[str]) -> List[dict]:
    persona_id = persona_id or None
    return [e for e in entries if memory_persona(e) == persona_id]


def tag_entry(entry: dict, persona_id: Optional[str]) -> dict:
    if persona_id:
        entry["persona"] = persona_id
    return entry


def remove_persona_memories(memory_manager, memory_vector, persona_id: str, owner: Optional[str]) -> int:
    """Delete every memory scoped to ``persona_id`` (and its vector entries)."""
    if not persona_id:
        return 0
    entries = memory_manager.load_all_for_update()
    doomed = [
        e for e in entries
        if memory_persona(e) == persona_id and (owner is None or e.get("owner") in (owner, None))
    ]
    if not doomed:
        return 0
    doomed_ids = {e.get("id") for e in doomed}
    memory_manager.save([e for e in entries if e.get("id") not in doomed_ids])
    if memory_vector is not None and getattr(memory_vector, "healthy", False):
        for mid in doomed_ids:
            try:
                memory_vector.remove(mid)
            except Exception:
                logger.debug("vector delete failed for %s", mid, exc_info=True)
    return len(doomed)
