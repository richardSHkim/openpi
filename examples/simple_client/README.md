# Simple Client

A minimal client that sends observations to the server and prints the inference rate.

You can specify which runtime environment to use using the `--env` flag. You can see the available options by running:

```bash
uv run examples/simple_client/main.py --help
```

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/simple_client/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
uv run examples/simple_client/main.py --env DROID
```

Terminal window 2:

```bash
uv run scripts/serve_policy.py --env DROID
```

For a custom Piper checkpoint, start the server with checkpoint mode and then point the client at the `PIPER` environment:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=checkpoints/pi05_piper/<exp>/<step>
```

```bash
uv run examples/simple_client/main.py --env PIPER
```
