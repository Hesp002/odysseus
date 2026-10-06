"""Per-action approval policy for tools that touch the host filesystem or run code.

The external-context gate in ``tool_capabilities`` only arms once untrusted
web/MCP content has entered a run. This policy applies to every call of the
code-running and file-writing tools, whether or not that has happened.

Modes (``ODYSSEUS_TOOL_APPROVAL_MODE``):

* ``off`` (default): no policy; upstream behavior.
* ``auto``: run routine work, ask before risky actions, block a few actions
  outright.
* ``ask``: ask before every write or command except plainly read-only ones.
  The block list still applies.

Pattern rules over shell text are a seatbelt, not a sandbox: a determined
model can hide a command inside ``python -c`` or an encoded string. The
container's mounts are the real boundary. These rules catch the routine
mistakes and the obvious injected instructions.

A ``block`` verdict cannot be approved. An ``ask`` verdict becomes an approval
card for that exact action. Approving it does not approve later actions.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from typing import Any, Iterable

MODE_ENV = "ODYSSEUS_TOOL_APPROVAL_MODE"
MODES = ("off", "ask", "auto")

ALLOW = "allow"
ASK = "ask"
BLOCK = "block"

OUTBOX_HINT = (
    "~/.config/hypr and ~/.config/omarchy are read-only here, because changes "
    "there run code on the host. Write the change to the host outbox instead "
    "(see the host-outbox skill) and the user will review and run it."
)


@dataclass(frozen=True)
class PolicyVerdict:
    level: str
    reason: str | None = None


_ALLOWED = PolicyVerdict(ALLOW)


def approval_mode() -> str:
    mode = (os.getenv(MODE_ENV) or "off").strip().lower()
    return mode if mode in MODES else "off"


def policy_active() -> bool:
    """Whether the per-action policy guards every host-touching call.

    When it does, locally configured context (the skills catalogue,
    integration and MCP tool descriptions) is still kept out of the system
    role but no longer arms the external-context gate: it is present on
    every turn, so arming on it would turn every action into an approval.
    Real outside content (web results, fetched pages, email, MCP tool
    results) still arms it.
    """
    return approval_mode() != "off"


# --- Always blocked -------------------------------------------------------

# Credential stores. Matched anywhere in a command or path.
_SECRET_RE = re.compile(
    r"""(
        (^|[\s/'"=:~])\.ssh(/|\b)
      | \.gnupg\b
      | \.config/gh\b
      | \.git-credentials\b
      | \.netrc\b
      | \bkeyrings?\b
      | \.password-store\b
      | \.mozilla\b
      | BraveSoftware
      | \.config/chromium\b
      | google-chrome
      | \.config/Proton\b
      | \.docker/config\.json
      | \.aws/
      | \.kube/
      | \bid_(rsa|dsa|ecdsa|ed25519)\b
      | /etc/(shadow|gshadow|sudoers)\b
      | \bapp\.key\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# The policy itself, the app's own code, and its auth/settings state.
_SELF_RE = re.compile(
    r"""(
        \bcommand_policy\b
      | \bTOOL_APPROVAL_MODE\b
      | \btool_(approvals?|approval_scopes|capabilities|security)\b
      | /app/(src|routes|core|services|static|app\.py|setup\.py|docker)\b
      | \bapp\.db\b
      | \bauth\.json\b
      | data/settings\.json\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_HOST_CONFIG_RE = re.compile(r"\.config/(hypr|omarchy)(/|\b)")

_CURL_SHORT_UPLOAD_RE = re.compile(r"\s-[a-zA-Z]*[dFT](\s|['\"@{]|$)")
_UPLOAD_LONG_RE = re.compile(
    r"""(
        \s--(data|data-raw|data-binary|data-urlencode|form|form-string|upload-file|json)\b
      | \s(-X|--request)\s*['"]?(POST|PUT|PATCH|DELETE)\b
      | \s--post-(data|file)\b
      | \s--body-(data|file)\b
      | \s--method[=\s]['"]?(POST|PUT|PATCH|DELETE)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)
# A download or decoded blob piped into a shell, or into an interpreter that
# reads its program from stdin (`| python3 -m json.tool` is fine).
_PIPE_TO_SHELL_RE = re.compile(
    r"\b(curl|wget|base64|xxd)\b[^\n;]*\|\s*(sudo\s+)?(env\s+)?"
    r"((ba|z|da|fi|k)?sh\b|source\b|\.\s"
    r"|(python[0-9.]*|perl|ruby|node|php)\s*(-\s*)?($|[;&|\n)]))"
)
_SHELL_FROM_DOWNLOAD_RE = re.compile(
    r"\b((ba|z|da|fi|k)?sh|source|\.|python[0-9.]*|perl|ruby|node)\s+<\(\s*(curl|wget)\b"
)

# Utilities whose path arguments are all written to (or removed).
_WRITES_EVERY_ARG = {"rm", "rmdir", "touch", "mkdir", "chmod", "chown", "truncate", "tee", "patch", "shred", "unlink", "mv"}
# Utilities that write only to their last argument.
_WRITES_LAST_ARG = {"cp", "install", "ln", "rsync"}


# --- Asked ---------------------------------------------------------------

_PREFIX_WORDS = {"command", "exec", "nohup", "time", "nice", "env", "xargs", "timeout", "stdbuf"}
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

_PRIVILEGE = {"sudo", "su", "doas", "pkexec", "run0"}
_REMOTE = {"ssh", "scp", "sftp", "nc", "ncat", "netcat", "socat", "telnet", "ftp", "mosh"}
_SYSTEM = {
    "systemctl", "docker", "podman", "kill", "pkill", "killall", "reboot",
    "shutdown", "poweroff", "mount", "umount", "mkfs", "fdisk", "parted",
    "crontab", "chown", "eval",
}
_DELETE = {"rm", "rmdir", "shred", "unlink"}
_SCRATCH_PREFIXES = ("/tmp/", "/var/tmp/")

_GIT_ALWAYS_ASK = {"push", "commit", "rebase", "filter-branch", "filter-repo", "clean", "am"}
_PKG_SUBCOMMANDS = {
    "pip": {"install", "uninstall", "download"},
    "pip3": {"install", "uninstall", "download"},
    "pipx": {"install", "inject", "upgrade", "uninstall"},
    "npm": {"install", "i", "add", "ci", "uninstall", "remove", "rm", "update", "exec", "x"},
    "pnpm": {"install", "i", "add", "remove", "update", "dlx"},
    "yarn": {"install", "add", "remove", "upgrade", "dlx"},
    "bun": {"install", "i", "add", "remove", "update", "x"},
    "apt": {"install", "remove", "purge", "upgrade", "full-upgrade", "autoremove"},
    "apt-get": {"install", "remove", "purge", "upgrade", "dist-upgrade", "autoremove"},
    "dnf": {"install", "remove", "upgrade", "update"},
    "cargo": {"install", "uninstall"},
    "gem": {"install", "uninstall"},
    "go": {"install", "get"},
    "brew": {"install", "uninstall", "upgrade"},
    "flatpak": {"install", "uninstall", "update"},
    "snap": {"install", "remove", "refresh"},
    "conda": {"install", "remove", "update"},
    "mamba": {"install", "remove", "update"},
}
_PKG_ALWAYS = {"npx", "pacman", "yay", "paru", "dpkg", "rpm", "apk", "uvx", "pip-sync"}

# Commands that cannot change anything; used only by "ask" mode.
_READ_ONLY = {
    "ls", "cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg",
    "find", "fd", "wc", "pwd", "file", "stat", "du", "df", "tree", "echo",
    "printf", "which", "type", "whoami", "id", "date", "uname", "env",
    "printenv", "sort", "uniq", "cut", "tr", "diff", "cmp", "jq", "realpath",
    "dirname", "basename", "readlink", "md5sum", "sha256sum", "column", "nl",
    "true", "false", "test", "[", "coredumpctl", "journalctl", "ps", "free",
    "uptime", "lscpu", "lsblk", "hostname", "zstdcat", "zcat", "xxd", "od",
    "strings", "nm", "objdump", "readelf", "ldd",
}
_GIT_READ_ONLY = {
    "status", "diff", "log", "show", "blame", "branch", "remote", "rev-parse",
    "ls-files", "grep", "describe", "shortlog", "reflog", "config", "tag",
    "fetch", "stash",
}

_PY_SIDE_EFFECT_RE = re.compile(
    r"\b(subprocess|os\.(system|popen|exec\w*|spawn\w*|remove|unlink|rmdir|removedirs|kill)"
    r"|shutil\.(rmtree|move)|requests|httpx|urllib|aiohttp|socket|http\.client|ftplib|smtplib"
    r"|pty|ctypes)\b"
)


def _ask(reason: str) -> PolicyVerdict:
    return PolicyVerdict(ASK, reason)


def _block(reason: str) -> PolicyVerdict:
    return PolicyVerdict(BLOCK, reason)


def _strictest(verdicts: Iterable[PolicyVerdict]) -> PolicyVerdict:
    asked = None
    for verdict in verdicts:
        if verdict.level == BLOCK:
            return verdict
        if verdict.level == ASK and asked is None:
            asked = verdict
    return asked or _ALLOWED


def _decode(content: Any, keys: tuple[str, ...]) -> str:
    """Return the command/code text from a raw or JSON-wrapped tool input."""
    if isinstance(content, dict):
        payload = content
    else:
        text = str(content or "")
        stripped = text.strip()
        if not stripped.startswith("{"):
            return text
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError):
            return text
        if not isinstance(payload, dict):
            return text
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(payload)


def _segments(command: str) -> list[str]:
    """Split shell text into simple commands, honouring quotes.

    Splits on ; & | newlines and parentheses outside quotes, and on $( and
    backticks everywhere except inside single quotes, so substituted commands
    are checked as commands of their own.
    """
    segments: list[str] = []
    current: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if quote == "'":
            current.append(ch)
            if ch == "'":
                quote = ""
        elif ch == "\\" and i + 1 < len(command):
            current.append(command[i:i + 2])
            i += 1
        elif ch == "`" or (ch == "$" and command[i + 1:i + 2] == "("):
            segments.append("".join(current))
            current = []
            if ch == "$":
                i += 1
        elif quote == '"':
            current.append(ch)
            if ch == '"':
                quote = ""
        elif ch in "'\"":
            quote = ch
            current.append(ch)
        elif ch in ";&|\n()":
            if ch == "&" and current and current[-1] in (">", "<"):
                current.append(ch)  # 2>&1, >&2
            else:
                segments.append("".join(current))
                current = []
        else:
            current.append(ch)
        i += 1
    segments.append("".join(current))
    return [seg.strip() for seg in segments if seg.strip()]


def _redirect_targets(segment: str) -> list[str]:
    return re.findall(r"(?<![0-9&<])>>?\|?\s*([^\s;&|<>]+)", segment) + re.findall(
        r"[0-9]>>?\s*([^\s;&|<>&][^\s;&|<>]*)", segment
    )


def _writes_host_config(segment: str, words: list[str]) -> bool:
    if not _HOST_CONFIG_RE.search(segment):
        return False
    if any(_HOST_CONFIG_RE.search(t) for t in _redirect_targets(segment)):
        return True
    if not words:
        return False
    head = os.path.basename(words[0])
    paths = [w for w in words[1:] if not w.startswith("-")]
    if head == "sed" and any(w.startswith("-i") or w == "--in-place" for w in words[1:]):
        return any(_HOST_CONFIG_RE.search(p) for p in paths)
    if head in _WRITES_EVERY_ARG:
        return any(_HOST_CONFIG_RE.search(p) for p in paths)
    if head in _WRITES_LAST_ARG and paths:
        return bool(_HOST_CONFIG_RE.search(paths[-1]))
    if head == "git" and "-C" in words:
        return True
    return False


def _words(segment: str) -> list[str]:
    try:
        return shlex.split(segment, comments=True)
    except ValueError:
        return segment.split()


def _command_words(segment: str) -> list[str]:
    """Strip env assignments and wrappers (sudo, nohup, ...) off a segment."""
    words = _words(segment)
    while words:
        head = os.path.basename(words[0])
        if _ASSIGNMENT_RE.match(words[0]):
            words = words[1:]
        elif head in _PRIVILEGE:
            # Keep the privilege wrapper as the command so it is always asked.
            break
        elif head in _PREFIX_WORDS:
            words = words[1:]
            # Drop the wrapper's own flags and arguments (timeout 30, nice -n 5).
            while words and (words[0].startswith("-") or words[0].isdigit()):
                words = words[1:]
        else:
            break
    return words


def _shell_blocks(text: str) -> PolicyVerdict | None:
    if _SECRET_RE.search(text):
        return _block("This touches a credential store (SSH/GPG keys, gh or git credentials, keyrings, browser profiles).")
    if _SELF_RE.search(text):
        return _block("This touches Odysseus's own code, approval policy, or auth/settings state.")
    if _PIPE_TO_SHELL_RE.search(text) or _SHELL_FROM_DOWNLOAD_RE.search(text):
        return _block("Piping a download or decoded data straight into an interpreter is not allowed. Save it, show it, and ask.")
    for segment in _segments(text):
        words = _command_words(segment)
        head = os.path.basename(words[0]) if words else ""
        if head in ("curl", "wget") and (
            _UPLOAD_LONG_RE.search(" " + segment)
            or (head == "curl" and _CURL_SHORT_UPLOAD_RE.search(" " + segment))
        ):
            return _block("curl/wget may only download here. Sending data out (POST/PUT, -d, -F, -T, --post-*) is not allowed.")
        if _writes_host_config(segment, words):
            return _block(OUTBOX_HINT)
    return None


def _segment_verdict(words: list[str]) -> PolicyVerdict:
    if not words:
        return _ALLOWED
    head = os.path.basename(words[0])
    args = words[1:]

    if head in _PRIVILEGE:
        return _ask(f"`{head}` runs with elevated privileges.")
    if head in _REMOTE:
        return _ask(f"`{head}` connects to another machine.")
    if head in ("curl", "wget"):
        return _ask("Downloads need approval. Use web_fetch to read pages.")
    if head == "rsync" and any(":" in arg and not arg.startswith("-") for arg in args):
        return _ask("rsync to or from another machine.")
    if head in _SYSTEM:
        return _ask(f"`{head}` changes system or process state.")
    if head in ("sh", "bash", "zsh", "dash") and "-c" in args:
        return _ask("Nested shell commands need approval so they can be reviewed.")
    if head == "git":
        return _git_verdict(args)
    if head in _DELETE or head == "truncate":
        targets = [a for a in args if not a.startswith("-")]
        if not targets or not all(t.startswith(_SCRATCH_PREFIXES) for t in targets):
            return _ask(f"`{head}` outside /tmp needs approval.")
        return _ALLOWED
    if head == "find" and any(a in ("-delete", "-exec", "-execdir", "-ok") for a in args):
        return _ask("find with -delete/-exec needs approval.")
    if head == "dd":
        return _ask("dd can overwrite files and devices.")
    if head in _PKG_ALWAYS:
        return _ask(f"`{head}` installs or runs packages.")
    if head == "uv":
        if args[:2] == ["pip", "install"] or (args and args[0] in ("add", "remove", "sync", "tool")):
            return _ask("Package install/removal needs approval.")
        return _ALLOWED
    if re.match(r"python[0-9.]*$", head) and args[:2] == ["-m", "pip"]:
        if len(args) > 2 and args[2] in _PKG_SUBCOMMANDS["pip"]:
            return _ask("Package install/removal needs approval.")
        return _ALLOWED
    subcommands = _PKG_SUBCOMMANDS.get(head)
    if subcommands and args and args[0] in subcommands:
        return _ask("Package install/removal needs approval.")
    return _ALLOWED


def _git_verdict(args: list[str]) -> PolicyVerdict:
    # Skip global options such as -C <dir> or -c key=value.
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-C", "-c", "--git-dir", "--work-tree") else 1
    if i >= len(args):
        return _ALLOWED
    sub, rest = args[i], args[i + 1:]
    if sub in _GIT_ALWAYS_ASK:
        return _ask(f"`git {sub}` needs approval.")
    forced = any(a in ("-f", "--force", "-D", "--hard", "--force-with-lease") or a.startswith("--force") for a in rest)
    if sub == "reset" and "--hard" in rest:
        return _ask("`git reset --hard` discards work.")
    if sub == "stash" and rest[:1] and rest[0] in ("drop", "clear"):
        return _ask("Dropping stashes discards work.")
    if sub in ("checkout", "switch", "branch", "tag", "update-ref", "restore") and forced:
        return _ask(f"Forced `git {sub}` can discard work.")
    if sub == "update-ref" and "-d" in rest:
        return _ask("Deleting refs needs approval.")
    return _ALLOWED


def _read_only_segment(words: list[str]) -> bool:
    if not words:
        return True
    head = os.path.basename(words[0])
    if head == "git":
        sub = next((w for w in words[1:] if not w.startswith("-")), "")
        return sub in _GIT_READ_ONLY and _git_verdict(words[1:]).level == ALLOW
    if head == "find":
        return not any(a in ("-delete", "-exec", "-execdir", "-ok", "-fprint") for a in words[1:])
    if head == "sed":
        return not any(a.startswith("-i") for a in words[1:])
    return head in _READ_ONLY


def evaluate_shell(command: str, mode: str) -> PolicyVerdict:
    blocked = _shell_blocks(command)
    if blocked:
        return blocked
    segments = [_command_words(s) for s in _segments(command)]
    verdict = _strictest(_segment_verdict(words) for words in segments)
    if verdict.level != ALLOW or mode != "ask":
        return verdict
    if re.search(r"(^|[^&2])>(?!&)", command.replace("2>/dev/null", "").replace(">/dev/null", "")):
        return _ask("Ask mode: this command writes output to a file.")
    if all(_read_only_segment(words) for words in segments):
        return _ALLOWED
    return _ask("Ask mode: commands that are not plainly read-only need approval.")


def evaluate_python(code: str, mode: str) -> PolicyVerdict:
    blocked = _shell_blocks(code)
    if blocked:
        return blocked
    if _HOST_CONFIG_RE.search(code) and re.search(r"open\([^)]*['\"][wax+]|write_text|write_bytes|shutil|os\.(rename|replace)", code):
        return _block(OUTBOX_HINT)
    if mode == "ask":
        return _ask("Ask mode: running code needs approval.")
    if _PY_SIDE_EFFECT_RE.search(code):
        return _ask("This code runs commands, deletes files, or uses the network.")
    return _ALLOWED


_PATCH_PATH_RE = re.compile(
    r"^(?:\*\*\* (?:Add|Update|Delete) File:|\*\*\* Move to:|\+\+\+ |--- )\s*(?:[ab]/)?(\S+)",
    re.MULTILINE,
)


def _write_paths(tool_name: str, content: Any) -> list[str]:
    raw = content if isinstance(content, str) else json.dumps(content or {})
    stripped = raw.strip()
    payload: Any = content if isinstance(content, dict) else None
    if payload is None and stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError):
            payload = None
    if tool_name == "apply_patch":
        text = raw
        if isinstance(payload, dict):
            text = str(payload.get("patch") or payload.get("input") or payload.get("content") or raw)
        return [p for p in _PATCH_PATH_RE.findall(text) if p != "/dev/null"]
    if isinstance(payload, dict):
        path = payload.get("path") or payload.get("file_path") or payload.get("file")
        return [str(path)] if path else []
    first = stripped.split("\n", 1)[0].strip()
    return [first] if first else []


def evaluate_write(tool_name: str, content: Any, mode: str) -> PolicyVerdict:
    paths = _write_paths(tool_name, content)
    verdicts = []
    for path in paths:
        expanded = os.path.expanduser(path)
        if _SECRET_RE.search(" " + expanded):
            verdicts.append(_block(f"{path} is in a credential store."))
        elif _SELF_RE.search(expanded):
            verdicts.append(_block(f"{path} is Odysseus's own code or state."))
        elif _HOST_CONFIG_RE.search(expanded):
            verdicts.append(_block(OUTBOX_HINT))
    verdict = _strictest(verdicts)
    if verdict.level == ALLOW and mode == "ask":
        return _ask("Ask mode: file changes need approval.")
    return verdict


def _evaluate_bg_jobs(content: Any, mode: str) -> PolicyVerdict:
    text = _decode(content, ("command", "cmd"))
    payload: Any = content if isinstance(content, dict) else None
    if payload is None:
        try:
            payload = json.loads(str(content or "").strip() or "{}")
        except (TypeError, ValueError):
            payload = None
    action = str((payload or {}).get("action") or "").lower() if isinstance(payload, dict) else ""
    if action and action not in ("start", "run", "spawn"):
        return _ALLOWED
    return evaluate_shell(text, mode)


_REMOTE_HEADS = _REMOTE | {"curl", "wget", "rsync"}


def trusts_workspace_result(tool_name: Any, content: Any, mode: str | None = None) -> bool:
    """Whether a local command/file result may skip arming the external gate.

    With the policy on, every later risky action is asked or blocked anyway,
    so output from the agent's own commands and files no longer forces an
    approval for everything after it. Commands that pull content from another
    machine (curl, wget, ssh, ...) still arm it: that output is external.
    """
    mode = mode or approval_mode()
    if mode == "off" or not isinstance(tool_name, str):
        return False
    if tool_name in ("write_file", "edit_file", "apply_patch", "read_file", "ls", "glob", "grep"):
        return True
    if tool_name == "python":
        return not _PY_SIDE_EFFECT_RE.search(_decode(content, ("code", "command")))
    if tool_name in ("bash", "manage_bg_jobs"):
        text = _decode(content, ("command", "cmd", "code"))
        for segment in _segments(text):
            words = _command_words(segment)
            heads = {os.path.basename(w) for w in words[:2]}  # sudo curl, ...
            if heads & _REMOTE_HEADS:
                return False
        return True
    return False


def evaluate(tool_name: Any, content: Any, mode: str | None = None) -> PolicyVerdict:
    """Classify one tool action. Unlisted tools are always allowed here."""
    mode = mode or approval_mode()
    if mode == "off" or not isinstance(tool_name, str):
        return _ALLOWED
    if tool_name == "bash":
        return evaluate_shell(_decode(content, ("command", "cmd", "code")), mode)
    if tool_name == "python":
        return evaluate_python(_decode(content, ("code", "command")), mode)
    if tool_name == "manage_bg_jobs":
        return _evaluate_bg_jobs(content, mode)
    if tool_name in ("write_file", "edit_file", "apply_patch"):
        return evaluate_write(tool_name, content, mode)
    return _ALLOWED
