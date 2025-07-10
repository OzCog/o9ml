#!/bin/bash
BUILD_DIR=$1

# Ensure the correct GHC version is installed for the resolver
echo "Setting up Stack environment with resolver-specified GHC version..."
stack setup --allow-different-user

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/opencog"

libname=$(cat opencog-lojban-wrapper.cabal | awk 'NR==1' | sed 's/name://g' | sed 's/ //g')
libver=$(cat opencog-lojban-wrapper.cabal | awk 'NR==2' | sed 's/version: //g' | sed "s/ //g")


if [ "$(id -u)" -ne 0 ];
then
    # Build haskell bindings package.
    stack build --allow-different-user
fi

LIB=$(find . -name "*$libname*.so" | awk 'NR==1')

patchelf --set-soname "lib$libname-$libver.so" $LIB

cp $LIB "$BUILD_DIR/lib$libname-$libver.so"
