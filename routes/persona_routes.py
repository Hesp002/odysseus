"""Persona routes — list/create/edit/delete chat personas, and read a chat's persona.

See src/personas.py. "Odysseus" (the default) is not stored; it is returned
first in the list with ``id: null`` so the UI can render it like the others.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src import personas
from src.auth_helpers import get_current_user

logger = logging.getLogger(__name__)


class PersonaRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    personality: str = Field(default="", max_length=8000)


class PersonaUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    personality: Optional[str] = Field(default=None, max_length=8000)


def setup_persona_routes(session_manager, memory_manager, memory_vector=None) -> APIRouter:
    router = APIRouter(prefix="/api/personas", tags=["personas"])

    def _owner(request: Request) -> Optional[str]:
        return get_current_user(request)

    def _require_privilege(request: Request):
        from src.auth_helpers import require_privilege
        require_privilege(request, "can_manage_memory")

    @router.get("")
    def api_list_personas(request: Request):
        default = {"id": None, "name": personas.DEFAULT_PERSONA_NAME, "personality": "", "default": True}
        return {"personas": [default] + personas.list_personas(_owner(request))}

    @router.post("")
    def api_create_persona(request: Request, body: PersonaRequest):
        _require_privilege(request)
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "Persona name is required")
        if name.lower() == personas.DEFAULT_PERSONA_NAME.lower():
            raise HTTPException(400, f"{personas.DEFAULT_PERSONA_NAME} is the default persona")
        return personas.create_persona(_owner(request), name, body.personality.strip())

    @router.put("/{persona_id}")
    def api_update_persona(request: Request, persona_id: str, body: PersonaUpdateRequest):
        _require_privilege(request)
        name = body.name.strip() if body.name is not None else None
        if name is not None and name.lower() == personas.DEFAULT_PERSONA_NAME.lower():
            raise HTTPException(400, f"{personas.DEFAULT_PERSONA_NAME} is the default persona")
        updated = personas.update_persona(
            persona_id,
            _owner(request),
            name=name,
            personality=body.personality.strip() if body.personality is not None else None,
        )
        if not updated:
            raise HTTPException(404, "Persona not found")
        return updated

    @router.delete("/{persona_id}")
    def api_delete_persona(request: Request, persona_id: str):
        _require_privilege(request)
        owner = _owner(request)
        if not personas.get_persona(persona_id, owner):
            raise HTTPException(404, "Persona not found")
        removed = personas.remove_persona_memories(memory_manager, memory_vector, persona_id, owner)
        personas.delete_persona(persona_id, owner)
        return {"ok": True, "memories_removed": removed}

    @router.get("/session/{session_id}")
    def api_session_persona(request: Request, session_id: str):
        owner = _owner(request)
        try:
            sess = session_manager.get_session(session_id)
        except KeyError:
            raise HTTPException(404, "Session not found")
        if owner and getattr(sess, "owner", None) not in (owner, None):
            raise HTTPException(404, "Session not found")
        persona = personas.session_persona(session_id)
        return {"persona": persona}

    return router
