import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import releaseConfig from './release.config.js';

test('release notes render through semantic-release with the configured preset', async () => {
  const notesPlugin = releaseConfig.plugins.find(
    (plugin) => Array.isArray(plugin) && plugin[0] === '@semantic-release/release-notes-generator',
  );
  assert.ok(notesPlugin, 'release configuration must include the notes generator');

  const require = createRequire(import.meta.url);
  const semanticReleaseRequire = createRequire(require.resolve('semantic-release'));
  const { generateNotes } = await import(pathToFileURL(
    semanticReleaseRequire.resolve('@semantic-release/release-notes-generator'),
  ));

  const featureHash = '0123456789abcdef0123456789abcdef01234567';
  const fixHash = 'abcdef0123456789abcdef0123456789abcdef01';
  const breakingHash = '123456789abcdef0123456789abcdef012345678';
  const repositoryUrl = 'https://github.com/cmeister2/git-facade';
  const notes = await generateNotes(notesPlugin[1], {
    cwd: fileURLToPath(new URL('.', import.meta.url)),
    env: process.env,
    options: { repositoryUrl },
    lastRelease: { version: '1.4.2', gitTag: '1.4.2' },
    nextRelease: { version: '2.0.0', gitTag: '2.0.0' },
    commits: [
      { hash: featureHash, message: 'feat: support prefix previews' },
      { hash: fixHash, message: 'fix: preserve signed commit headers\n\nCloses #123' },
      {
        hash: breakingHash,
        message: 'feat!: require hexadecimal prefixes\n\nBREAKING CHANGE: Prefixes must contain hexadecimal digits.',
      },
    ],
    logger: { log() {} },
  });

  assert.match(notes, /^### Features$/m);
  assert.match(notes, /^### Bug Fixes$/m);
  assert.match(notes, /^### .*BREAKING CHANGES$/m);
  assert.ok(notes.includes('support prefix previews'));
  assert.ok(notes.includes('preserve signed commit headers'));
  assert.ok(notes.includes('require hexadecimal prefixes'));
  assert.ok(notes.includes('Prefixes must contain hexadecimal digits.'));
  assert.ok(notes.includes(`${repositoryUrl}/compare/1.4.2...2.0.0`));
  assert.ok(notes.includes(`${repositoryUrl}/issues/123`));
  for (const hash of [featureHash, fixHash, breakingHash]) {
    assert.ok(notes.includes(`${repositoryUrl}/commit/${hash}`));
  }
});
