#!/bin/bash
set -euo pipefail

NEW_VERSION=$1
WGPU_CRATE=wgpu-sha1
WGPU_MANIFEST=wgpu-sha1/Cargo.toml

read_manifest_version() {
	local manifest=$1

	sed -n 's/^version = "\(.*\)"/\1/p' "$manifest" | head -n 1
}

configure_git_identity() {
	git config user.name "github-actions[bot]"
	git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
}

push_tag() {
	local tag=$1

	if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
		git tag -d "$tag"
	fi

	configure_git_identity
	git tag "$tag"
	git push origin "refs/tags/${tag}"
}

crate_version_exists() {
	local crate=$1
	local version=$2

	curl --silent --show-error --fail "https://crates.io/api/v1/crates/${crate}/${version}" >/dev/null 2>&1
}

wait_for_crate_version() {
	local crate=$1
	local version=$2
	local attempt

	for attempt in $(seq 1 30); do
		if crate_version_exists "$crate" "$version"; then
			return 0
		fi

		echo "waiting for ${crate} ${version} to appear on crates.io (${attempt}/30)"
		sleep 10
	done

	echo "timed out waiting for ${crate} ${version} to appear on crates.io" >&2
	return 1
}

publish_wgpu_sha1_if_needed() {
	local version tag

	version=$(read_manifest_version "$WGPU_MANIFEST")
	if [ -z "$version" ]; then
		echo "failed to read ${WGPU_CRATE} version from ${WGPU_MANIFEST}" >&2
		exit 1
	fi

	tag="${WGPU_CRATE}-v${version}"

	if git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
		echo "remote tag ${tag} already exists; skipping ${WGPU_CRATE} publish"
		return 0
	fi

	if crate_version_exists "$WGPU_CRATE" "$version"; then
		echo "${WGPU_CRATE} ${version} already exists on crates.io; backfilling ${tag}"
		push_tag "$tag"
		return 0
	fi

	echo "publishing ${WGPU_CRATE} ${version} before git-facade"
	cargo publish -p "$WGPU_CRATE" --allow-dirty
	wait_for_crate_version "$WGPU_CRATE" "$version"
	push_tag "$tag"
}

publish_wgpu_sha1_if_needed

# Update the version in git-facade/Cargo.toml
sed -i "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" git-facade/Cargo.toml

# Publish git-facade to crates.io after its dependency is available.
cargo publish -p git-facade --allow-dirty

# Set outputs for downstream jobs
if [ -n "${GITHUB_OUTPUT:-}" ]; then
	echo "new_release=true" >> "$GITHUB_OUTPUT"
	echo "version=${NEW_VERSION}" >> "$GITHUB_OUTPUT"
fi
