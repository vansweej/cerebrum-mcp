{
  description = "Cerebrum: two-tier agent memory subsystem";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain = pkgs.rust-bin.stable.latest.default;

        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

        cerebrum = rustPlatform.buildRustPackage {
          pname = "cerebrum";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = with pkgs; [ pkg-config protobuf ];
          buildInputs = with pkgs; [ ];

          cargoBuildFlags = [ "-p" "cerebrum" ];
          doCheck = false;
        };
      in
      {
        # `cerebrum` self-locates its data directory in-binary (see
        # crates/cerebrum-core/src/config.rs's `default_data_dir`, which
        # resolves XDG_DATA_HOME/HOME to the same path the former wrapper's
        # `cd` produced), so no wrapper script is needed to set cwd anymore.
        packages.default = cerebrum;
        packages.cerebrum = cerebrum;

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            cargo
            rustfmt
            clippy
            cargo-tarpaulin
            just
            pkg-config
            openssl
            protobuf
          ];

          shellHook = ''
            echo "Cerebrum development environment loaded"
          '';
        };
      }
    );
}
