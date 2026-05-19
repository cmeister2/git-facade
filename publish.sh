#!/bin/bash
set -euxo pipefail
NEW_VERSION=$1

# Update the version in git-facade/Cargo.toml
sed -i "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" git-facade/Cargo.toml

# Publish git-facade to crates.io
cargo publish -p git-facade --allow-dirty

# Set outputs for downstream jobs
echo "new_release=true" >> $GITHUB_OUTPUT
echo "version=${NEW_VERSION}" >> $GITHUB_OUTPUT
