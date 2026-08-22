"""One-shot: install this machine's public key on a password-only host.

Kept out of the training path; delete once the host trusts the key.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko


def main() -> int:
    host = os.environ["BOOTSTRAP_HOST"]
    port = int(os.environ["BOOTSTRAP_PORT"])
    user = os.environ.get("BOOTSTRAP_USER", "root")
    password = os.environ["BOOTSTRAP_PASSWORD"]
    pubkey = Path(os.environ.get(
        "BOOTSTRAP_PUBKEY", str(Path.home() / ".ssh/id_ed25519.pub")
    )).read_text().strip()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password,
                   timeout=30, allow_agent=False, look_for_keys=False)
    command = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"grep -qxF '{pubkey}' ~/.ssh/authorized_keys 2>/dev/null || "
        f"echo '{pubkey}' >> ~/.ssh/authorized_keys; "
        "chmod 600 ~/.ssh/authorized_keys && echo KEY_INSTALLED"
    )
    _, stdout, stderr = client.exec_command(command, timeout=30)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    client.close()
    print(out or err)
    return 0 if "KEY_INSTALLED" in out else 1


if __name__ == "__main__":
    sys.exit(main())
