# Development

Use a dedicated Conda environment named `apexveil` and run commands from this repository. Runtime assets belong in `models/`; evaluation input belongs in `data/private/`; generated work belongs in `outputs/`.

After installation, run `python -m pytest -q` and `ruff check src tests --select E9,F63,F7,F82`. CPU CI never downloads weights or requires a GPU. Run GPU smoke tests explicitly on one selected device and keep the report local until it has been checked for private data.

Configure any required network proxy in the shell that installs dependencies; do not hard-code a host's proxy, SSH address, or account into source. Hugging Face authentication stays in the operator's normal credential store.

For a future GUI listening on port 7860:

```sh
ssh -N -L 127.0.0.1:7860:127.0.0.1:7860 your-gpu-host
```

Then open http://127.0.0.1:7860. The SSH connection is independent of ApexVeil. The editing server itself is on the roadmap and is not part of this foundation release.
