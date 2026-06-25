# Cerebrum

A two-tier agent memory subsystem implemented as a single Model Context Protocol (MCP) server.

## Quick Start

```bash
nix develop . --command cargo run --bin cerebrum
```

## Development

All commands should be run inside the Nix dev shell:

```bash
nix develop . --command cargo fmt
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo test
nix develop . --command cargo tarpaulin --out Html --output-dir coverage
```

Or enter the dev shell once and run commands directly:

```bash
nix develop
cargo fmt
cargo clippy -- -D warnings
cargo test
cargo tarpaulin --out Html --output-dir coverage
```

## Code Quality Requirements

- **Coverage Gate:** All code must maintain ≥90% test coverage (configured in `tarpaulin.toml`, enforced by `cargo tarpaulin`)
- **Formatting:** Code must be formatted with `cargo fmt`
- **Linting:** All clippy warnings must be fixed (run `cargo clippy -- -D warnings`)

## Architecture

See [docs/architecture.md](docs/architecture.md) for system design and memory tier documentation.

## License

MIT
