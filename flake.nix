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
          buildInputs = with pkgs; [ openssl ];

          cargoBuildFlags = [ "-p" "cerebrum" ];
          doCheck = false;
        };

        cerebrum-wrapped = pkgs.writeShellApplication {
          name = "cerebrum";
          runtimeInputs = [ cerebrum ];
          text = ''
            DATA_DIR="''${XDG_DATA_HOME:-$HOME/.local/share}/cerebrum"
            mkdir -p "$DATA_DIR"
            cd "$DATA_DIR"
            exec cerebrum "$@"
          '';
        };
      in
      {
        packages.default = cerebrum-wrapped;
        packages.cerebrum = cerebrum;
        packages.cerebrum-wrapped = cerebrum-wrapped;

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
