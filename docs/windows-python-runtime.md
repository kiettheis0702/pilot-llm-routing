# Windows Python runtime repair — 2026-10-30

> Historical runtime-repair record. SDK login and upload were completed later;
> W&B is now Public. See [current handoff](CLAUDE_HANDOFF.md). The “login pending”
> statements below describe the state immediately after the runtime repair.

The user reported `No Python at ...` when starting W&B from ordinary CMD.
Both `.venv` and `.venv-embedding` referenced a Python 3.11.18 installation
under `%APPDATA%\uv\python`. Inside packaged Codex that path resolved to
`AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Roaming\uv\python`.
The nominal AppData path did not exist in an ordinary desktop process.
This is Windows AppData virtualization, not a missing W&B package.

[Microsoft documents the AppData redirection](https://learn.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-behind-the-scenes#appdata-operations-on-windows-10-version-1903-and-later).
[uv documents that moving its Python installation does not update existing environments](https://docs.astral.sh/uv/reference/storage/#python-versions).

## Applied repair

- Installed the same Python 3.11.18 under
  `%USERPROFILE%\.local\share\uv\python`, outside the redirected AppData tree.
- Rebuilt both environments' launchers with `uv venv --allow-existing`, keeping
  installed packages. All 54 main and 31 embedding package versions matched
  the inventories saved before repair.
- Repaired the `python3.11.exe` command with uv's installer. The default
  system Python was not changed.
- Pinned the absolute local interpreter with `uv python pin`; `.python-version`
  is already ignored by this repository. Do not copy that machine-specific pin
  or virtual environments to another PC.
- Saved original configs, launchers, package inventories and diagnostic probes
  under `C:\Users\ADMIN\.codex\tmp\wandb-runtime-20261030`.

For a recurrence on this workstation, these PowerShell commands use the same
version and shared location. Ensure no Python jobs are using these environments
and back up their configs/launchers first. Never use `--clear` here.

```powershell
$runtimeDir = Join-Path $env:USERPROFILE '.local\share\uv\python'
uv python install 3.11.18 --install-dir $runtimeDir --no-registry --force
$runtimeExe = Join-Path $runtimeDir 'cpython-3.11.18-windows-x86_64-none\python.exe'
uv venv --allow-existing --python $runtimeExe --no-python-downloads --prompt pilot-llm-routing .venv
uv venv --allow-existing --python $runtimeExe --no-python-downloads .venv-embedding
uv python pin $runtimeExe
uv sync --locked --extra cost --extra tracking --inexact --check
uv pip check --python .venv\Scripts\python.exe
```

## Verification and limits

An ordinary desktop CMD was launched hidden with the desktop process as its
parent, without elevated privileges. Its Python process reported no package
identity and confirmed that the old nominal AppData path did not exist.
A copy of the old launcher/config reproduced the user's exact `No Python at`
error with exit 103. The repaired main Python, embedding Python,
`python3.11.exe`, `wandb --version`, and
`wandb login --cloud --verify --help` all exited 0. Merely using a child CMD of
Codex did not reproduce the error; it inherited the redirected filesystem.
The final evidence is `external-result.json` in the diagnostic directory above.

`uv sync --locked --extra cost --extra tracking --inexact --check` reported no
changes needed. `uv pip check` reported compatible dependencies. The complete
serial regression passed: **472 passed, one Windows symlink test skipped**.
The first default-temp invocation had 311 passes and 162 fixture setup errors
because `AppData\Local\Temp\pytest-of-ADMIN` denied access. Running with a new
`--basetemp` under the project's `.cache` resolved those setup errors; no ACLs
were changed. The passing output is saved as `pytest.txt` in the diagnostic
directory. Native math thread counts were limited to one.

SDK account authentication and online uploads were not performed: the user
still needs to enter the API key locally. No paid API request was made.

## Handoff

- Trạng thái: XONG — sửa lỗi khởi chạy Python/W&B; đăng nhập tài khoản chờ key.
- Gate: locked environment check, dependency check, 472 tests passed / 1 skipped.
- E2E: ordinary desktop CMD reproduced the old failure and passed the repaired CLI checks.
- Probe: six external-process checks passed, including the expected old-launcher failure.
- Đường hỏng: old launcher exits 103 with the exact reported `No Python at` error.
- Người thứ hai: n/a — single-user CLI; verified outside the agent's app context.
- Lưu trữ: reread both pyvenv.cfg files, local pin, and unchanged package inventories.
- Chưa chạy: actual account login and online upload; no API key supplied to the CLI.
- Bug đã biết: default pytest temp directory denies access; use a fresh project-local basetemp.
- Quyết định: keep the same Python and package versions; change runtime location only.
- Tôi sẽ lo: runtime repair and local verification are complete.
- Cần bạn: run the login command and paste the key only into its local prompt.

## Lesson

Source: user feedback “fix triệt để đi”, chat
`01a18c8b-7663-78e2-9567-19140e0998ab`, AI pilot, 2026-10-30.
For CLI handoffs from packaged Windows apps, verify both shell syntax and the
interpreter's physical location, then run the command under the user's normal
desktop context. A successful existence check or an agent-owned terminal is
not sufficient evidence that the user's CMD can start the same interpreter.
