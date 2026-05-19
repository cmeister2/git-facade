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
    _pad0: u32,
    _pad1: u32,
}

struct FindResult {
    found: atomic<u32>,
    salt_lo: u32,
    salt_hi: u32,
}

@group(0) @binding(0) var<storage, read> template_data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> prefix_data: array<u32>;
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
    var w: array<u32, 80>;

    // Load the 16 message words
    for (var i = 0u; i < 16u; i++) {
        w[i] = (*block)[i];
    }

    // Expand to 80 words
    for (var i = 16u; i < 80u; i++) {
        w[i] = rotl(w[i - 3u] ^ w[i - 8u] ^ w[i - 14u] ^ w[i - 16u], 1u);
    }

    var a = (*state)[0];
    var b = (*state)[1];
    var c = (*state)[2];
    var d = (*state)[3];
    var e = (*state)[4];

    // Rounds 0-19
    for (var i = 0u; i < 20u; i++) {
        let temp = rotl(a, 5u) + sha1_f0(b, c, d) + e + 0x5A827999u + w[i];
        e = d;
        d = c;
        c = rotl(b, 30u);
        b = a;
        a = temp;
    }

    // Rounds 20-39
    for (var i = 20u; i < 40u; i++) {
        let temp = rotl(a, 5u) + sha1_f1(b, c, d) + e + 0x6ED9EBA1u + w[i];
        e = d;
        d = c;
        c = rotl(b, 30u);
        b = a;
        a = temp;
    }

    // Rounds 40-59
    for (var i = 40u; i < 60u; i++) {
        let temp = rotl(a, 5u) + sha1_f2(b, c, d) + e + 0x8F1BBCDCu + w[i];
        e = d;
        d = c;
        c = rotl(b, 30u);
        b = a;
        a = temp;
    }

    // Rounds 60-79
    for (var i = 60u; i < 80u; i++) {
        let temp = rotl(a, 5u) + sha1_f1(b, c, d) + e + 0xCA62C1D6u + w[i];
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

// Hex lookup table as an array
const HEX_TABLE: array<u32, 16> = array<u32, 16>(
    0x30u, 0x31u, 0x32u, 0x33u, 0x34u, 0x35u, 0x36u, 0x37u,
    0x38u, 0x39u, 0x61u, 0x62u, 0x63u, 0x64u, 0x65u, 0x66u
);

// Get a byte from the template data (stored as big-endian u32 words)
fn get_template_byte(byte_idx: u32) -> u32 {
    let word_idx = byte_idx / 4u;
    let byte_pos = byte_idx % 4u;
    let word = template_data[word_idx];
    // Bytes stored in big-endian order within each u32
    return (word >> (24u - byte_pos * 8u)) & 0xFFu;
}

// Write the hex encoding of a u64 salt (16 ASCII hex chars) into a byte array at offset
fn write_salt_bytes(data: ptr<function, array<u32, 256>>, salt_lo: u32, salt_hi: u32, offset: u32) {
    // 16 hex chars: hi word provides chars 0-7, lo word provides chars 8-15
    for (var i = 0u; i < 8u; i++) {
        let nibble = (salt_hi >> (28u - i * 4u)) & 0xFu;
        let byte_idx = offset + i;
        let word_idx = byte_idx / 4u;
        let byte_pos = byte_idx % 4u;
        let shift = 24u - byte_pos * 8u;
        (*data)[word_idx] = ((*data)[word_idx] & ~(0xFFu << shift)) | (HEX_TABLE[nibble] << shift);
    }
    for (var i = 0u; i < 8u; i++) {
        let nibble = (salt_lo >> (28u - i * 4u)) & 0xFu;
        let byte_idx = offset + 8u + i;
        let word_idx = byte_idx / 4u;
        let byte_pos = byte_idx % 4u;
        let shift = 24u - byte_pos * 8u;
        (*data)[word_idx] = ((*data)[word_idx] & ~(0xFFu << shift)) | (HEX_TABLE[nibble] << shift);
    }
}

fn sha1_of_template(salt_lo: u32, salt_hi: u32) -> array<u32, 5> {
    let total_bytes = params.template_len_bytes;

    // Copy template into private memory as big-endian u32 words
    let num_words = (total_bytes + 3u) / 4u;
    var data: array<u32, 256>; // max 1024 bytes
    for (var i = 0u; i < num_words; i++) {
        data[i] = template_data[i];
    }

    // Write salt hex encoding into the template
    write_salt_bytes(&data, salt_lo, salt_hi, params.salt_offset_bytes);

    // SHA1 init
    var state: array<u32, 5> = array<u32, 5>(
        0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u
    );

    // Process complete 64-byte blocks
    let full_blocks = total_bytes / 64u;
    for (var b = 0u; b < full_blocks; b++) {
        var block: array<u32, 16>;
        let base = b * 16u;
        for (var i = 0u; i < 16u; i++) {
            block[i] = data[base + i];
        }
        sha1_compress(&state, &block);
    }

    // Handle the final (possibly partial) block + padding
    let remaining = total_bytes - full_blocks * 64u;
    let remaining_words_start = full_blocks * 16u;

    // Build the padded final block(s)
    // We need to append: 0x80 byte, zeros, 64-bit big-endian bit count
    // If remaining <= 55, fits in one block. Otherwise need two blocks.
    var final_block: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        final_block[i] = 0u;
    }

    // Copy remaining bytes (as u32 words)
    let remaining_full_words = remaining / 4u;
    for (var i = 0u; i < remaining_full_words; i++) {
        final_block[i] = data[remaining_words_start + i];
    }

    // Handle partial last word + 0x80 byte
    let leftover_bytes = remaining % 4u;
    if leftover_bytes == 0u {
        // 0x80 goes at the start of the next word
        final_block[remaining_full_words] = 0x80000000u;
    } else {
        // Copy the partial word and insert 0x80
        var partial = data[remaining_words_start + remaining_full_words];
        let mask = 0xFFFFFFFFu << ((4u - leftover_bytes) * 8u);
        partial = partial & mask;
        partial = partial | (0x80u << ((3u - leftover_bytes) * 8u));
        final_block[remaining_full_words] = partial;
    }

    let bit_count = total_bytes * 8u;

    if remaining <= 55u {
        // Everything fits in one block
        final_block[15] = bit_count;
        sha1_compress(&state, &final_block);
    } else {
        // Need two blocks
        sha1_compress(&state, &final_block);
        var extra_block: array<u32, 16>;
        for (var i = 0u; i < 16u; i++) {
            extra_block[i] = 0u;
        }
        extra_block[15] = bit_count;
        sha1_compress(&state, &extra_block);
    }

    return state;
}

fn check_prefix(state: array<u32, 5>) -> bool {
    let prefix_len = params.prefix_len;
    // Compare full u32 words first
    let full_words = prefix_len / 4u;
    for (var i = 0u; i < full_words; i++) {
        if state[i] != prefix_data[i] {
            return false;
        }
    }
    // Compare remaining bytes
    let leftover = prefix_len % 4u;
    if leftover > 0u {
        let mask = 0xFFFFFFFFu << ((4u - leftover) * 8u);
        if (state[full_words] & mask) != (prefix_data[full_words] & mask) {
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

    let state = sha1_of_template(salt_lo, salt_hi);

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

    // Salt is passed as pairs in prefix_data for this entry point:
    // prefix_data[idx*2] = salt_lo, prefix_data[idx*2+1] = salt_hi
    let salt_lo = prefix_data[idx * 2u];
    let salt_hi = prefix_data[idx * 2u + 1u];

    let state = sha1_of_template(salt_lo, salt_hi);

    let base = idx * 5u;
    debug_digests[base + 0u] = state[0];
    debug_digests[base + 1u] = state[1];
    debug_digests[base + 2u] = state[2];
    debug_digests[base + 3u] = state[3];
    debug_digests[base + 4u] = state[4];
}
