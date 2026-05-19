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
    _pad1: u32,
}

struct FindResult {
    found: atomic<u32>,
    salt_lo: u32,
    salt_hi: u32,
}

@group(0) @binding(0) var<storage, read> template_data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> aux_data: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: FindResult;
@group(0) @binding(4) var<storage, read_write> debug_digests: array<u32>;

fn rotl(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

fn sha1_f0(b: u32, c: u32, d: u32) -> u32 {
    return (b & c) | ((~b) & d);
}

fn sha1_f1(b: u32, c: u32, d: u32) -> u32 {
    return b ^ c ^ d;
}

fn sha1_f2(b: u32, c: u32, d: u32) -> u32 {
    return (b & c) | (b & d) | (c & d);
}

fn sha1_compress(state: ptr<function, array<u32, 5>>, block: ptr<function, array<u32, 16>>) {
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

fn schedule_word(schedule: ptr<function, array<u32, 16>>, round: u32) -> u32 {
    let slot = round & 15u;
    if round >= 16u {
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

// Hex lookup table as an array
const HEX_TABLE: array<u32, 16> = array<u32, 16>(
    0x30u, 0x31u, 0x32u, 0x33u, 0x34u, 0x35u, 0x36u, 0x37u,
    0x38u, 0x39u, 0x61u, 0x62u, 0x63u, 0x64u, 0x65u, 0x66u
);

fn salt_hex_byte(salt_lo: u32, salt_hi: u32, idx: u32) -> u32 {
    if idx < 8u {
        let nibble = (salt_hi >> (28u - idx * 4u)) & 0xFu;
        return HEX_TABLE[nibble];
    }

    let nibble = (salt_lo >> (28u - (idx - 8u) * 4u)) & 0xFu;
    return HEX_TABLE[nibble];
}

fn write_block_byte(block: ptr<function, array<u32, 16>>, byte_idx: u32, value: u32) {
    let word_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    let shift = 24u - byte_pos * 8u;
    (*block)[word_idx] = ((*block)[word_idx] & ~(0xFFu << shift)) | ((value & 0xFFu) << shift);
}

fn patch_salt_in_block(block: ptr<function, array<u32, 16>>, block_start: u32, salt_lo: u32, salt_hi: u32) {
    for (var i = 0u; i < 16u; i++) {
        let salt_pos = params.salt_offset_bytes + i;
        if salt_pos >= block_start && salt_pos < block_start + 64u {
            write_block_byte(block, salt_pos - block_start, salt_hex_byte(salt_lo, salt_hi, i));
        }
    }
}

fn load_prefix_state(offset: u32) -> array<u32, 5> {
    return array<u32, 5>(
        aux_data[offset + 0u],
        aux_data[offset + 1u],
        aux_data[offset + 2u],
        aux_data[offset + 3u],
        aux_data[offset + 4u]
    );
}

fn sha1_of_template(salt_lo: u32, salt_hi: u32, prefix_state_offset: u32) -> array<u32, 5> {
    let suffix_bytes = params.template_len_bytes;
    var state = load_prefix_state(prefix_state_offset);

    let full_blocks = suffix_bytes / 64u;
    for (var b = 0u; b < full_blocks; b++) {
        var block: array<u32, 16>;
        let base = b * 16u;
        for (var i = 0u; i < 16u; i++) {
            block[i] = template_data[base + i];
        }
        patch_salt_in_block(&block, b * 64u, salt_lo, salt_hi);
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
        let mask = 0xFFFFFFFFu << ((4u - leftover_bytes) * 8u);
        final_block[remaining_full_words] = partial & mask;
    }

    patch_salt_in_block(&final_block, full_blocks * 64u, salt_lo, salt_hi);
    let padding_shift = 24u - leftover_bytes * 8u;
    final_block[remaining_full_words] = final_block[remaining_full_words] | (0x80u << padding_shift);

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

@compute @workgroup_size(256)
fn find_prefix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.batch_size {
        return;
    }

    // Check if already found
    if atomicLoad(&result.found) != 0u {
        return;
    }

    // Compute salt = salt_base + idx as u64
    let salt_lo = params.salt_base_lo + idx;
    var salt_hi = params.salt_base_hi;
    if salt_lo < params.salt_base_lo {
        salt_hi += 1u;  // carry
    }

    let prefix_state_offset = (params.prefix_len + 3u) / 4u;
    let state = sha1_of_template(salt_lo, salt_hi, prefix_state_offset);

    if check_prefix(state) {
        // Atomic CAS to claim the result
        let old = atomicCompareExchangeWeak(&result.found, 0u, 1u);
        if old.exchanged {
            result.salt_lo = salt_lo;
            result.salt_hi = salt_hi;
        }
    }
}

@compute @workgroup_size(1)
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
