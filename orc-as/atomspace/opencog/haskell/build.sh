#!/bin/sh
BIN_DIR=$1

# Ensure the correct GHC version is installed for the resolver
echo "Setting up Stack environment with resolver-specified GHC version..."
stack setup --allow-different-user

if [ "$(id -u)" -ne 0 ];
then
  # Build haskell bindings package.
  stack build --allow-different-user --extra-lib-dirs=${BIN_DIR}
fi
