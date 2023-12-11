{
  description = "HOPS implementation of the TU-Dresden Theoretical Quantum Optics Group";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils.url = "github:vale981/hiro-flake-utils";
    utils.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, utils, nixpkgs, ... }:
    (utils.lib.poetry2nixWrapper nixpkgs {
      name = "multi-hops";
      shellPackages = pkgs: with pkgs; [ pyright python39Packages.jupyter redis ];
      poetryArgs = {
        projectDir = ./.;
      };
      python = pkgs: pkgs.python39;
      shellOverride = _: _: {
        shellHook = ''
                  export PYTHONPATH=$(realpath ./)
                  '';
      };
    });
}
