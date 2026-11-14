"""Kit launch arguments shared by RoboLab entry points.

Isaac Sim creates an inotify file-change watch for every extension directory it
loads. On Linux the per-user watch limit (fs.inotify.max_user_watches) is often
only a few thousand, so a large local extension cache exhausts it and Kit logs
thousands of "Failed to create change watch ...: errno=28" errors. Training and
evaluation never need extension hot reload, so the watcher can be disabled for
the process through Kit's own setting.
"""

from __future__ import annotations

DISABLE_EXTENSION_FS_WATCHER = "--/app/extensions/fsWatcherEnabled=false"
_FS_WATCHER_SETTING = "--/app/extensions/fsWatcherEnabled"


def disable_extension_fs_watcher(kit_args: str | None) -> str:
    """Return kit_args with the extension file watcher disabled exactly once.

    A caller-supplied --/app/extensions/fsWatcherEnabled=... value is kept as-is:
    the helper must not silently override an explicit choice. The returned
    string is passed to Isaac Lab's --kit_args argument, which splits it into
    tokens before forwarding them to Kit.
    """
    tokens = (kit_args or "").split()
    if any(token == _FS_WATCHER_SETTING or token.startswith(_FS_WATCHER_SETTING + "=")
           for token in tokens):
        return " ".join(tokens)
    tokens.append(DISABLE_EXTENSION_FS_WATCHER)
    return " ".join(tokens)
