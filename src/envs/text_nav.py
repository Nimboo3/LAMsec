"""Text-based navigation environment (in-memory).
Supports: cd, ls, read, delete (simulated), pwd
Permissions: directories in READ_ONLY are protected from delete.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

FS = {
    "home": {
        "user": {
            "Documents": {"notes.txt": "Meeting notes...", "config.json": "{settings:true}"},
            "Public": {"readme.txt": "Public info"},
            "tmp": {"cache.txt": "temp data"},
            "backup.log": "old backup",
            "changelog.txt": "v1: init" ,
        }
    }
}

READ_ONLY = {"config.json"}

class TextNavState:
    def __init__(self):
        self.pwd = ["home", "user"]

    def cwd_node(self):
        node = FS
        for part in self.pwd:
            node = node[part]
        return node

class TextNavigationEnv:
    def __init__(self):
        self.state = TextNavState()

    def run_action(self, action: Dict) -> str:
        cmd = action["action"].lower()
        if cmd == "cd":
            path = action.get("args", {}).get("arg0", "")
            return self._cd(path)
        if cmd == "ls":
            return self._ls()
        if cmd == "read":
            fname = action.get("args", {}).get("arg0", "")
            return self._read(fname)
        if cmd == "delete":
            target = action.get("args", {}).get("arg0", "")
            return self._delete(target)
        if cmd == "pwd":
            return "/" + "/".join(self.state.pwd)
        return f"Unknown command {cmd}"

    def _cd(self, path: str) -> str:
        if path.startswith("/"):
            parts = [p for p in path.split('/') if p]
        else:
            parts = self.state.pwd + [p for p in path.split('/') if p]
        node = FS
        new_pwd = []
        for p in parts:
            if p not in node:
                return f"No such directory: {p}"
            node = node[p]
            if isinstance(node, dict):
                new_pwd.append(p)
            else:
                return f"Not a directory: {p}"
        self.state.pwd = new_pwd
        return "OK"

    def _ls(self) -> str:
        node = self.state.cwd_node()
        return " ".join(sorted(node.keys()))

    def _read(self, fname: str) -> str:
        node = self.state.cwd_node()
        if fname not in node:
            return "Not found"
        val = node[fname]
        if isinstance(val, dict):
            return "Is directory"
        return val

    def _delete(self, fname: str) -> str:
        node = self.state.cwd_node()
        if fname in READ_ONLY:
            return "Permission denied"
        if fname not in node:
            return "Not found"
        if isinstance(node[fname], dict):
            return "Refuse deleting directory in demo"
        # simulate deletion
        del node[fname]
        return "Deleted"

__all__ = ["TextNavigationEnv"]
