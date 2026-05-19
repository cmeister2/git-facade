#!/bin/bash
set -euxo pipefail
NEW_VERSION=$1

# Update the version in Cargo.toml
sed -i "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" Cargo.toml

# Publish to crates.io
cargo publish --allow-dirty

# Set outputs for downstream jobs
echo "new_release=true" >> $GITHUB_OUTPUT
echo "version=${NEW_VERSION}" >> $GITHUB_OUTPUT
