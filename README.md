# git-facade

A Rust port of [gitc0ffee](https://github.com/trichner/gitc0ffee) - a tool to find "vanity" hashes for git commits. Make all your commit hashes start with `c0ffee`, `facade`, `cafe`, `badc0de5`, or whatever makes you happy!

## Install

```bash
cargo install --path .
```

## Usage

```bash
# do a normal git commit
git commit -am '...'

# update the last commit with a vanity hash
git-facade --update-ref --prefix c0ffee
```

## Options

| Flag           | Default      | Description                                                   |
|----------------|--------------|---------------------------------------------------------------|
| `--prefix`     | `c0ffee`     | Hex prefix to brute-force (even-length, lowercase `[0-9a-f]`) |
| `--solver`     | `concurrent` | Solver to use: `concurrent` or `singlethreaded`               |
| `--update-ref` | `false`      | Update HEAD to point to the new commit object                 |

## How it works

This is a faithful port of the [Go implementation](https://github.com/trichner/gitc0ffee) to Rust:

1. Read the latest commit digest (`git rev-parse HEAD`).
2. Parse the raw commit object (`git cat-file -p <digest>`).
3. Add a `coffeesalt` header to the commit object and brute-force the salt value until the SHA1 hash starts with the desired prefix.
4. Write the new commit object to the git store (`git hash-object -w -t commit --stdin`).
5. Optionally update the current branch to the new commit (`git update-ref HEAD <new digest>`).

## Performance

The concurrent solver uses [rayon](https://github.com/rayon-rs/rayon) to parallelise across all available CPU cores.

- 6-character prefix: under a second
- 8-character prefix: minutes

Prefixes beyond 8 characters may not finish in useful time.

## Prefix ideas

All even-length hexadecimal prefixes work (`[0-9a-f]{2,40}`). For inspiration, see [Hexspeak](https://en.wikipedia.org/wiki/Hexspeak):

`facade`, `c0ffee`, `cafe`, `badc0de`, `deadbeef`, `0ff1ce`, `dec0de`, `defaced`

## Credits

Original Go implementation by [Thomas Richner](https://github.com/trichner/gitc0ffee). This Rust port aims for full algorithmic parity with the original.
