{ nixpkgs ? import <nixpkgs> {} }:
let nixpkgs_source = fetchTarball https://github.com/NixOS/nixpkgs/archive/4cf0b6ba5d5ab5eb20a88449e0612f4dad8e4c29.tar.gz;
in

with (import nixpkgs_source {}).pkgs;

(python.buildEnv.override {
  extraLibs =  with python36Packages; [tensorflowWithCuda Keras nltk matplotlib notebook h5py scipy gensim pillow ] ;
  ignoreCollisions = true;
}).env

