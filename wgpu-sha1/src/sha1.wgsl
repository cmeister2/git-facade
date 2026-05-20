// SHA1 compute shader for GPU-accelerated digest computation.
//
// Two entry points:
//   find_prefix  — production: checks if SHA1(template with salt) starts with a prefix
//   compute_digest — testing: writes full SHA1 digest to output buffer

struct Params {
    salt_offset_bytes: u32,
    template_len_bytes: u32,
    prefix_len: u32,
    batch_size: u32,
    salt_base_lo: u32,
    salt_base_hi: u32,
    total_len_bytes: u32,
    dispatch_groups_x: u32,
}

struct FindResult {
    found: atomic<u32>,
    salt_lo: u32,
    salt_hi: u32,
}

@group(0) @binding(0) var<storage, read> template_data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
// Auxiliary read-only data. For find_prefix this stores prefix words followed by
// the five-word SHA1 prefix state; for compute_digest it stores salt pairs
// followed by the same prefix state.
@group(0) @binding(2) var<storage, read> aux_data: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: FindResult;
@group(0) @binding(4) var<storage, read_write> debug_digests: array<u32>;

// Rotate left, used by SHA1 rounds and message schedule expansion.
fn rotl(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

// SHA1 choose function for rounds 0..19.
fn sha1_f0(b: u32, c: u32, d: u32) -> u32 {
    return (b & c) | ((~b) & d);
}

// SHA1 parity function for rounds 20..39 and 60..79.
fn sha1_f1(b: u32, c: u32, d: u32) -> u32 {
    return b ^ c ^ d;
}

// SHA1 majority function for rounds 40..59.
fn sha1_f2(b: u32, c: u32, d: u32) -> u32 {
    return (b & c) | (b & d) | (c & d);
}

// Compress one 512-bit SHA1 message block into the five-word chaining state.
fn sha1_compress(state: ptr<function, array<u32, 5>>, block: ptr<function, array<u32, 16>>) {
    // Keep only the rolling 16-word schedule window instead of a private W[80].
    var schedule: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        schedule[i] = (*block)[i];
    }

    var a = (*state)[0];
    var b = (*state)[1];
    var c = (*state)[2];
    var d = (*state)[3];
    var e = (*state)[4];

    for (var i = 0u; i < 20u; i++) {
        let wt = schedule_word(&schedule, i);
        let temp = rotl(a, 5u) + sha1_f0(b, c, d) + e + 0x5A827999u + wt;
        e = d;
        d = c;
        c = rotl(b, 30u);
        b = a;
        a = temp;
    }

    for (var i = 20u; i < 40u; i++) {
        let wt = schedule_word(&schedule, i);
        let temp = rotl(a, 5u) + sha1_f1(b, c, d) + e + 0x6ED9EBA1u + wt;
        e = d;
        d = c;
        c = rotl(b, 30u);
        b = a;
        a = temp;
    }

    for (var i = 40u; i < 60u; i++) {
        let wt = schedule_word(&schedule, i);
        let temp = rotl(a, 5u) + sha1_f2(b, c, d) + e + 0x8F1BBCDCu + wt;
        e = d;
        d = c;
        c = rotl(b, 30u);
        b = a;
        a = temp;
    }

    for (var i = 60u; i < 80u; i++) {
        let wt = schedule_word(&schedule, i);
        let temp = rotl(a, 5u) + sha1_f1(b, c, d) + e + 0xCA62C1D6u + wt;
        e = d;
        d = c;
        c = rotl(b, 30u);
        b = a;
        a = temp;
    }

    (*state)[0] += a;
    (*state)[1] += b;
    (*state)[2] += c;
    (*state)[3] += d;
    (*state)[4] += e;
}

// Return W[round], generating and storing it in the rolling 16-word schedule
// once the initial message words have been consumed.
fn schedule_word(schedule: ptr<function, array<u32, 16>>, round: u32) -> u32 {
    let slot = round & 15u;
    if round >= 16u {
        // `schedule[slot]` still holds W[t-16] at this point.
        (*schedule)[slot] = rotl(
            (*schedule)[(round - 3u) & 15u] ^
            (*schedule)[(round - 8u) & 15u] ^
            (*schedule)[(round - 14u) & 15u] ^
            (*schedule)[slot],
            1u
        );
    }
    return (*schedule)[slot];
}

// Convert a 4-bit nibble to lowercase ASCII hex.
fn hex_ascii(nibble: u32) -> u32 {
    if nibble < 10u {
        return 0x30u + nibble;
    }
    return 0x57u + nibble;
}

// Pack four ASCII hex characters from `value` into one big-endian u32 word.
fn ascii_hex_word(value: u32, high_shift: u32) -> u32 {
    // `high_shift` is 28 or 12, selecting either the high or low 16 bits.
    return (hex_ascii((value >> high_shift) & 0xFu) << 24u) |
        (hex_ascii((value >> (high_shift - 4u)) & 0xFu) << 16u) |
        (hex_ascii((value >> (high_shift - 8u)) & 0xFu) << 8u) |
        hex_ascii((value >> (high_shift - 12u)) & 0xFu);
}

// Convert the u64 salt split across hi/lo words into four big-endian ASCII
// hex words: hi high, hi low, lo high, lo low.
fn salt_ascii_words(salt_lo: u32, salt_hi: u32) -> array<u32, 4> {
    return array<u32, 4>(
        ascii_hex_word(salt_hi, 28u),
        ascii_hex_word(salt_hi, 12u),
        ascii_hex_word(salt_lo, 28u),
        ascii_hex_word(salt_lo, 12u)
    );
}

// Write one byte into a big-endian u32 block word, preserving neighboring bytes.
fn write_block_byte(block: ptr<function, array<u32, 16>>, byte_idx: u32, value: u32) {
    let word_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    let shift = 24u - byte_pos * 8u;
    (*block)[word_idx] = ((*block)[word_idx] & ~(0xFFu << shift)) | ((value & 0xFFu) << shift);
}

// Patch one four-byte ASCII salt word into a local 64-byte SHA1 block.
fn patch_salt_word_in_block(block: ptr<function, array<u32, 16>>, block_start: u32, word_start: u32, salt_word: u32) {
    let block_end = block_start + 64u;
    if word_start >= block_end || word_start + 4u <= block_start {
        return;
    }

    if word_start >= block_start && word_start + 4u <= block_end {
        let local_byte = word_start - block_start;
        let word_idx = local_byte / 4u;
        let byte_offset = local_byte % 4u;

        if byte_offset == 0u {
            // Fast path: salt word is aligned with the destination message word.
            (*block)[word_idx] = salt_word;
        } else {
            // Unaligned path: split the big-endian salt word across two message
            // words without disturbing the non-salt bytes around it.
            let first_shift = byte_offset * 8u;
            let first_mask = 0xFFFFFFFFu >> first_shift;
            (*block)[word_idx] = ((*block)[word_idx] & ~first_mask) | ((salt_word >> first_shift) & first_mask);

            let second_shift = (4u - byte_offset) * 8u;
            let second_mask = 0xFFFFFFFFu << second_shift;
            (*block)[word_idx + 1u] = ((*block)[word_idx + 1u] & ~second_mask) | ((salt_word << second_shift) & second_mask);
        }
        return;
    }

    // Fallback for the rare case where this four-byte salt word crosses a
    // 64-byte SHA1 block boundary. Only overlapping bytes belong in this block.
    for (var i = 0u; i < 4u; i++) {
        let byte_pos = word_start + i;
        if byte_pos >= block_start && byte_pos < block_end {
            write_block_byte(block, byte_pos - block_start, (salt_word >> (24u - i * 8u)) & 0xFFu);
        }
    }
}

// Patch all four packed ASCII salt words that overlap the current block.
fn patch_salt_in_block(block: ptr<function, array<u32, 16>>, block_start: u32, salt_words: ptr<function, array<u32, 4>>) {
    for (var i = 0u; i < 4u; i++) {
        patch_salt_word_in_block(block, block_start, params.salt_offset_bytes + i * 4u, (*salt_words)[i]);
    }
}

// Load the precomputed SHA1 chaining state after all full prefix blocks.
fn load_prefix_state(offset: u32) -> array<u32, 5> {
    return array<u32, 5>(
        aux_data[offset + 0u],
        aux_data[offset + 1u],
        aux_data[offset + 2u],
        aux_data[offset + 3u],
        aux_data[offset + 4u]
    );
}

// Hash the suffix template with one candidate salt, starting from the CPU-made
// prefix state and appending standard SHA1 padding for the full original length.
fn sha1_of_template(salt_lo: u32, salt_hi: u32, prefix_state_offset: u32) -> array<u32, 5> {
    let suffix_bytes = params.template_len_bytes;
    var state = load_prefix_state(prefix_state_offset);
    var salt_words = salt_ascii_words(salt_lo, salt_hi);

    let full_blocks = suffix_bytes / 64u;
    for (var b = 0u; b < full_blocks; b++) {
        var block: array<u32, 16>;
        let base = b * 16u;
        for (var i = 0u; i < 16u; i++) {
            block[i] = template_data[base + i];
        }
        // Salt coordinates are suffix-relative, so each block patches only the
        // salt bytes/words that overlap its [block_start, block_start + 64) span.
        patch_salt_in_block(&block, b * 64u, &salt_words);
        sha1_compress(&state, &block);
    }

    let remaining = suffix_bytes - full_blocks * 64u;
    let remaining_words_start = full_blocks * 16u;

    var final_block: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        final_block[i] = 0u;
    }

    let remaining_full_words = remaining / 4u;
    for (var i = 0u; i < remaining_full_words; i++) {
        final_block[i] = template_data[remaining_words_start + i];
    }

    let leftover_bytes = remaining % 4u;
    if leftover_bytes != 0u {
        var partial = template_data[remaining_words_start + remaining_full_words];
        // Keep only real suffix bytes in the partial final word; padding is
        // inserted after salt patching so salt bytes are not overwritten.
        let mask = 0xFFFFFFFFu << ((4u - leftover_bytes) * 8u);
        final_block[remaining_full_words] = partial & mask;
    }

    patch_salt_in_block(&final_block, full_blocks * 64u, &salt_words);
    // Append the SHA1 0x80 padding byte at the first byte after the suffix.
    let padding_shift = 24u - leftover_bytes * 8u;
    final_block[remaining_full_words] = final_block[remaining_full_words] | (0x80u << padding_shift);

    // Full original message length, not suffix length, goes in the SHA1 length field.
    let bit_count_low = params.total_len_bytes << 3u;
    let bit_count_high = params.total_len_bytes >> 29u;

    if remaining <= 55u {
        final_block[14] = bit_count_high;
        final_block[15] = bit_count_low;
        sha1_compress(&state, &final_block);
    } else {
        sha1_compress(&state, &final_block);
        var extra_block: array<u32, 16>;
        for (var i = 0u; i < 16u; i++) {
            extra_block[i] = 0u;
        }
        extra_block[14] = bit_count_high;
        extra_block[15] = bit_count_low;
        sha1_compress(&state, &extra_block);
    }

    return state;
}

// Check whether the computed digest state starts with the requested byte prefix.
fn check_prefix(state: array<u32, 5>) -> bool {
    let prefix_len = params.prefix_len;
    // Compare full u32 words first
    let full_words = prefix_len / 4u;
    for (var i = 0u; i < full_words; i++) {
        if state[i] != aux_data[i] {
            return false;
        }
    }
    // Compare remaining bytes
    let leftover = prefix_len % 4u;
    if leftover > 0u {
        let mask = 0xFFFFFFFFu << ((4u - leftover) * 8u);
        if (state[full_words] & mask) != (aux_data[full_words] & mask) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
// Search one batch of consecutive salts for the first digest matching aux_data's prefix.
fn find_prefix(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The host may dispatch across x and y to stay under per-dimension dispatch
    // limits. Flatten back to one candidate index here.
    let idx = gid.x + gid.y * params.dispatch_groups_x * 64u;
    if idx >= params.batch_size {
        return;
    }

    if atomicLoad(&result.found) != 0u {
        return;
    }

    // Add the flattened candidate index to the u64 salt base.
    let salt_lo = params.salt_base_lo + idx;
    var salt_hi = params.salt_base_hi;
    if salt_lo < params.salt_base_lo {
        salt_hi += 1u;
    }

    let prefix_state_offset = (params.prefix_len + 3u) / 4u;
    let state = sha1_of_template(salt_lo, salt_hi, prefix_state_offset);

    if check_prefix(state) {
        let old = atomicCompareExchangeWeak(&result.found, 0u, 1u);
        if old.exchanged {
            result.salt_lo = salt_lo;
            result.salt_hi = salt_hi;
        }
    }
}

@compute @workgroup_size(1)
// Test-only entry point: compute full digests for explicit salt values in aux_data.
fn compute_digest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.batch_size {
        return;
    }

    let salt_lo = aux_data[idx * 2u];
    let salt_hi = aux_data[idx * 2u + 1u];

    let prefix_state_offset = params.batch_size * 2u;
    let state = sha1_of_template(salt_lo, salt_hi, prefix_state_offset);

    let base = idx * 5u;
    debug_digests[base + 0u] = state[0];
    debug_digests[base + 1u] = state[1];
    debug_digests[base + 2u] = state[2];
    debug_digests[base + 3u] = state[3];
    debug_digests[base + 4u] = state[4];
}
