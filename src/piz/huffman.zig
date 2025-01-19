//! This modules provides function for huffman encoding and decoding, specific
//! to the PIZ compression format.

const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;

const encoding_bits = 16;
const decoding_bits = 14;

const encoding_table_size = (1 << encoding_bits) + 1;
const decoding_table_size = 1 << decoding_bits;
const decoding_mask = decoding_table_size - 1;

const Encoding = packed struct {
    val: u58,
    len: u6,
};

const DecodingType = enum { empty, short, long };
const Decoding = union(DecodingType) {
    empty: void,
    short: ShortDecoding,
    long: std.ArrayList(u32),
};

const ShortDecoding = packed struct {
    val: u16,
    len: u6,
    _: u10 = 0,
};

fn countFrequencies(data: []const u16, allocator: std.mem.Allocator) ![]u64 {
    const frequencies = try allocator.alloc(u64, encoding_table_size);
    @memset(frequencies, 0);
    for (data) |value| {
        frequencies[value] += 1;
    }
    return frequencies;
}

test "count frequencies" {
    const allocator = std.testing.allocator;
    const data = [_]u16{ 3, 3, 3, 10, 5, 5, 1, 1, 1, 1 };
    const frequencies = try countFrequencies(&data, allocator);
    defer allocator.free(frequencies);
    try expect(frequencies[1] == 4);
    try expect(frequencies[3] == 3);
    try expect(frequencies[5] == 2);
    try expect(frequencies[10] == 1);
}

const Frequency = struct {
    frequency: u64,
    index: u32,

    fn compare(_: void, a: Frequency, b: Frequency) std.math.Order {
        const freq_order = std.math.order(a.frequency, b.frequency);
        switch (freq_order) {
            .eq => return std.math.order(a.index, b.index),
            else => return freq_order,
        }
    }
};
const FrequencyHeap = std.PriorityQueue(Frequency, void, Frequency.compare);

const EncodingTableResult = struct {
    encoding_table: []Encoding,
    min_code_idx: u32,
    max_code_idx: u32,
};

fn buildEncodingTable(
    frequencies: []const u64,
    allocator: std.mem.Allocator,
) !EncodingTableResult {
    assert(frequencies.len == encoding_table_size);
    // Initialize a heap of frequencies
    var heap = FrequencyHeap.init(allocator, {});
    defer heap.deinit();
    const link = try allocator.alloc(u32, encoding_table_size);
    defer allocator.free(link);
    var min_non_zero: u32 = encoding_table_size - 1;
    for (0..encoding_table_size) |i| {
        if (frequencies[i] == 0) continue;
        min_non_zero = @intCast(i);
        break;
    }
    var max_non_zero: u32 = 0;
    for (min_non_zero..encoding_table_size) |i| {
        link[i] = @intCast(i);
        if (frequencies[i] == 0) continue;
        try heap.add(.{ .frequency = frequencies[i], .index = @intCast(i) });
        max_non_zero = @intCast(i);
    }
    // Add a pseudo-symbol which will indicate run-length encoding
    try heap.add(.{ .frequency = 1, .index = max_non_zero + 1 });
    max_non_zero += 1;
    // Compute code lengths for each symbol
    const encoding_table = try allocator.alloc(Encoding, encoding_table_size);
    @memset(encoding_table, .{ .val = 0, .len = 0 });
    while (heap.count() > 1) {
        const lowest = heap.remove();
        const second_lowest = heap.remove();
        try heap.add(.{
            .frequency = lowest.frequency + second_lowest.frequency,
            .index = second_lowest.index,
        });
        var index = second_lowest.index;
        while (true) : (index = link[index]) {
            encoding_table[index].len += 1;
            assert(encoding_table[index].len < 59);
            if (index == link[index]) break;
        }
        link[index] = lowest.index;
        index = lowest.index;
        while (true) : (index = link[index]) {
            encoding_table[index].len += 1;
            assert(encoding_table[index].len < 59);
            if (index == link[index]) break;
        }
    }
    return .{
        .encoding_table = encoding_table,
        .min_code_idx = min_non_zero,
        .max_code_idx = max_non_zero,
    };
}

test "encoding lengths" {
    const allocator = std.testing.allocator;
    const data = [_]u16{
        1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5,
        5, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    };
    const frequencies = try countFrequencies(&data, allocator);
    defer allocator.free(frequencies);
    const result = try buildEncodingTable(frequencies, allocator);
    defer allocator.free(result.encoding_table);
    const encoding_table = result.encoding_table;
    const min_code_idx = result.min_code_idx;
    const max_code_idx = result.max_code_idx;
    // Only the symbols in our data should have code lengths.
    for (encoding_table[0..min_code_idx]) |encoding| {
        try expect(encoding.len == 0);
    }
    for (encoding_table[min_code_idx .. max_code_idx + 1]) |encoding| {
        try expect(encoding.len > 0);
    }
    for (encoding_table[max_code_idx + 1 ..]) |encoding| {
        try expect(encoding.len == 0);
    }
    // More frequent symbols shouldn't have longer codes.
    for (min_code_idx..max_code_idx + 1) |i| {
        for (min_code_idx..max_code_idx + 1) |j| {
            if (frequencies[i] <= frequencies[j]) continue;
            try expect(encoding_table[i].len <= encoding_table[j].len);
        }
    }
}

fn buildCanonicalEncodings(encoding_table: []Encoding) void {
    // This algorithm originates here: http://www.compressconsult.com/huffman/#huffman
    // How many codes of each length are there?
    var num_codes_per_length: [59]u32 = undefined;
    @memset(&num_codes_per_length, 0);
    for (encoding_table) |encoding| {
        num_codes_per_length[encoding.len] += 1;
    }
    // What is the lowest code for each length?
    var code: u58 = 0;
    var codes_per_length: [59]u58 = undefined;
    for (0..59) |i| {
        const j = 59 - (i + 1);
        codes_per_length[j] = code;
        code = (code + num_codes_per_length[j]) >> 1;
    }
    std.debug.assert(codes_per_length[0] == 1);
    // Assign lowest code, then increment.
    for (encoding_table) |*encoding| {
        if (encoding.len == 0) continue;
        encoding.val = codes_per_length[encoding.len];
        codes_per_length[encoding.len] += 1;
        std.debug.assert(encoding.val >> encoding.len == 0);
    }
}

test "canonical encodings" {
    const allocator = std.testing.allocator;
    const data = [_]u16{
        1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5,
        5, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    };
    const frequencies = try countFrequencies(&data, allocator);
    defer allocator.free(frequencies);
    const result = try buildEncodingTable(frequencies, allocator);
    defer allocator.free(result.encoding_table);
    const encoding_table = result.encoding_table;
    const min_code_idx = result.min_code_idx;
    const max_code_idx = result.max_code_idx;
    buildCanonicalEncodings(encoding_table);
    // Only the symbols in our data should have code lengths and values.
    // Code values should match the specified length.
    for (encoding_table[0..min_code_idx]) |encoding| {
        try expect(encoding.val == 0);
        try expect(encoding.len == 0);
    }
    for (encoding_table[min_code_idx .. max_code_idx + 1]) |encoding| {
        try expect(encoding.val >= 0);
        try expect(encoding.len > 0);
        try expect(encoding.val >> encoding.len == 0);
    }
    for (encoding_table[max_code_idx + 1 ..]) |encoding| {
        try expect(encoding.val == 0);
        try expect(encoding.len == 0);
    }
}

fn encode(
    data: []const u16,
    encoding_table: []const Encoding,
    rle_symbol: u32,
    writer: anytype,
) !u32 {
    var num_bits: u32 = 0;
    var bit_writer = std.io.bitWriter(std.builtin.Endian.big, writer);
    // Encode and write all symbols.
    const rle_encoding = encoding_table[rle_symbol];
    var run_length: u8 = 0;
    var run_symbol = data[0];
    for (data[1..]) |x| {
        if (x == run_symbol and run_length < 255) {
            run_length += 1;
        } else {
            const run_encoding = encoding_table[run_symbol];
            num_bits += try writeCode(
                run_encoding,
                rle_encoding,
                run_length,
                &bit_writer,
            );
            run_length = 0;
        }
        run_symbol = x;
    }
    const run_encoding = encoding_table[run_symbol];
    num_bits += try writeCode(
        run_encoding,
        rle_encoding,
        run_length,
        &bit_writer,
    );
    try bit_writer.flushBits();
    return num_bits;
}

fn writeCode(
    encoding: Encoding,
    rle_encoding: Encoding,
    run_length: u8,
    bit_writer: anytype,
) !u32 {
    var num_bits: u32 = 0;
    // Output `run_length` instances of `encoding`. This is either stored as
    // `encoding, rle_encoding, run_length` or `[encoding] * run_length`,
    // whichever is shortest.
    const rle_length = encoding.len + rle_encoding.len + 8;
    const rep_length = encoding.len * run_length;
    if (rle_length < rep_length) {
        try bit_writer.writeBits(encoding.val, encoding.len);
        try bit_writer.writeBits(rle_encoding.val, rle_encoding.len);
        try bit_writer.writeBits(run_length, 8);
        num_bits += encoding.len + rle_encoding.len + 8;
    } else {
        for (0..run_length + 1) |_| {
            try bit_writer.writeBits(encoding.val, encoding.len);
            num_bits += encoding.len;
        }
    }
    return num_bits;
}

test "encode" {
    // Adapted from tests in the 'exrs' project: https://github.com/johannesvollmer/exrs
    const allocator = std.testing.allocator;
    const uncompressed = [_]u16{
        3852,  2432,  33635, 49381, 10100, 15095, 62693, 63738, 62359, 5013,
        7715,  59875, 28182, 34449, 19983, 20399, 63407, 29486, 4877,  26738,
        44815, 14042, 46091, 48228, 25682, 35412, 7582,  65069, 6632,  54124,
        13798, 27503, 52154, 61961, 30474, 46880, 39097, 15754, 52897, 42371,
        54053, 14178, 48276, 34591, 42602, 32126, 42062, 31474, 16274, 55991,
        2882,  17039, 56389, 20835, 57057, 54081, 3414,  33957, 52584, 10222,
        25139, 40002, 44980, 1602,  48021, 19703, 6562,  61777, 41582, 201,
        31253, 51790, 15888, 40921, 3627,  12184, 16036, 26349, 3159,  29002,
        14535, 50632, 18118, 33583, 18878, 59470, 32835, 9347,  16991, 21303,
        26263, 8312,  14017, 41777, 43240, 3500,  60250, 52437, 45715, 61520,
    };
    const compressed = [_]u8{
        0x10, 0x9,  0xb4, 0xe4, 0x4c, 0xf7, 0xef, 0x42, 0x87, 0x6a, 0xb5, 0xc2,
        0x34, 0x9e, 0x2f, 0x12, 0xae, 0x21, 0x68, 0xf2, 0xa8, 0x74, 0x37, 0xe1,
        0x98, 0x14, 0x59, 0x57, 0x2c, 0x24, 0x3b, 0x35, 0x6c, 0x1b, 0x8b, 0xcc,
        0xe6, 0x13, 0x38, 0xc,  0x8e, 0xe2, 0xc,  0xfe, 0x49, 0x73, 0xbc, 0x2b,
        0x7b, 0x9,  0x27, 0x79, 0x14, 0xc,  0x94, 0x42, 0xf8, 0x7c, 0x1,  0x8d,
        0x26, 0xde, 0x87, 0x26, 0x71, 0x50, 0x45, 0xc6, 0x28, 0x40, 0xd5, 0xe,
        0x8d, 0x8,  0x1e, 0x4c, 0xa4, 0x79, 0x57, 0xf0, 0xc3, 0x6d, 0x5c, 0x6d,
        0xc0,
    };
    // Build encoding table for data.
    const frequencies = try countFrequencies(&uncompressed, allocator);
    defer allocator.free(frequencies);
    const result = try buildEncodingTable(frequencies, allocator);
    const encoding_table = result.encoding_table;
    const max_code_idx = result.max_code_idx;
    defer allocator.free(encoding_table);
    buildCanonicalEncodings(encoding_table);
    // Encode data into output buffer.
    var encoded = std.ArrayList(u8).init(allocator);
    defer encoded.deinit();
    const writer = encoded.writer();
    _ = try encode(&uncompressed, encoding_table, max_code_idx, writer);
    try expect(std.mem.eql(u8, encoded.items, &compressed));
}

fn buildDecodingTable(
    encoding_table: []const Encoding,
    min_code_idx: u32,
    max_code_idx: u32,
    allocator: std.mem.Allocator,
) ![]Decoding {
    var decoding_table = try allocator.alloc(Decoding, decoding_table_size);
    @memset(decoding_table, Decoding{ .empty = @as(void, {}) });
    for (min_code_idx..max_code_idx + 1) |encoding_idx| {
        const encoding = encoding_table[encoding_idx];
        if (encoding.val >> encoding.len != 0) {
            return error.InvalidEncodingTableEntry;
        }
        if (encoding.len > decoding_bits) {
            // If symbol `x` is encoded as `XX111111111111` and symbol `y` is
            // encoded as `YY111111111111`, then index `11111111111111` in the
            // decoding table will contain the list `[x, y]`.
            const shift = encoding.len - decoding_bits;
            const decoding_idx = encoding.val >> shift;
            const decoding = &decoding_table[decoding_idx];
            switch (decoding.*) {
                .empty => {
                    decoding.* = Decoding{
                        .long = std.ArrayList(u32).init(allocator),
                    };
                    try decoding.long.append(@intCast(encoding_idx));
                },
                .long => try decoding.long.append(@intCast(encoding_idx)),
                .short => return error.InvalidEncodingTableEntry,
            }
        } else if (encoding.len > 0) {
            // If symbol `x` is encoded as `XXX`, then all entries between
            // `XXX00000000000` and `XXX11111111111` map to the value `x` in
            // the decoding table.
            const shift = decoding_bits - encoding.len;
            const start: u64 = @as(u64, encoding.val) << shift;
            const count: u64 = @as(u64, 1) << shift;
            for (start..start + count) |i| {
                decoding_table[i] = Decoding{ .short = .{
                    .len = encoding.len,
                    .val = @intCast(encoding_idx),
                } };
            }
        }
    }
    return decoding_table;
}

test "build decoding table" {
    // Adapted from tests in the 'exrs' project: https://github.com/johannesvollmer/exrs
    const allocator = std.testing.allocator;
    const uncompressed = [_]u16{
        3852,  2432,  33635, 49381, 10100, 15095, 62693, 63738, 62359, 5013,
        7715,  59875, 28182, 34449, 19983, 20399, 63407, 29486, 4877,  26738,
        44815, 14042, 46091, 48228, 25682, 35412, 7582,  65069, 6632,  54124,
        13798, 27503, 52154, 61961, 30474, 46880, 39097, 15754, 52897, 42371,
        54053, 14178, 48276, 34591, 42602, 32126, 42062, 31474, 16274, 55991,
        2882,  17039, 56389, 20835, 57057, 54081, 3414,  33957, 52584, 10222,
        25139, 40002, 44980, 1602,  48021, 19703, 6562,  61777, 41582, 201,
        31253, 51790, 15888, 40921, 3627,  12184, 16036, 26349, 3159,  29002,
        14535, 50632, 18118, 33583, 18878, 59470, 32835, 9347,  16991, 21303,
        26263, 8312,  14017, 41777, 43240, 3500,  60250, 52437, 45715, 61520,
    };
    // Build encoding and decoding tables.
    const frequencies = try countFrequencies(&uncompressed, allocator);
    defer allocator.free(frequencies);
    const result = try buildEncodingTable(frequencies, allocator);
    defer allocator.free(result.encoding_table);
    const encoding_table = result.encoding_table;
    const min_code_idx = result.min_code_idx;
    const max_code_idx = result.max_code_idx;
    buildCanonicalEncodings(encoding_table);
    const decoding_table = try buildDecodingTable(
        encoding_table,
        min_code_idx,
        max_code_idx,
        allocator,
    );
    defer allocator.free(decoding_table);
    // Doesn't test long codes, but at least we know short codes work.
    for (uncompressed) |x| {
        const encoding = encoding_table[x];
        const decoding_idx = @as(u64, encoding.val) << (decoding_bits - encoding.len);
        const decoding = decoding_table[decoding_idx];
        try expect(decoding.short.val == x);
    }
}

const BitBufferError = error{
    BitOverflow,
    BitUnderflow,
};

const BitBuffer = struct {
    bits: u64 = 0,
    num_bits: u6 = 0,

    fn addBits(self: *@This(), bits: anytype, num_bits: u6) !void {
        if (@as(u8, 63) - num_bits < self.num_bits) {
            return error.BitOverflow;
        }
        self.num_bits += num_bits;
        const mask = (@as(u64, 1) << num_bits) - 1;
        self.bits = (self.bits << num_bits) | (bits & mask);
    }

    fn peekBits(self: *@This(), comptime T: type, num_bits: u6) !T {
        if (num_bits > self.num_bits) {
            return error.BitUnderflow;
        }
        const mask = (@as(u64, 1) << num_bits) - 1;
        const val = (self.bits >> (self.num_bits - num_bits)) & mask;
        return @as(T, @intCast(val));
    }

    fn readBits(self: *@This(), comptime T: type, num_bits: u6) !T {
        const val = try self.peekBits(T, num_bits);
        self.num_bits -= num_bits;
        return val;
    }

    fn dumpBits(self: *@This(), num_bits: u6) !void {
        if (num_bits > self.num_bits) {
            return error.BitUnderflow;
        }
        self.bits >>= @intCast(num_bits);
        self.num_bits -= @intCast(num_bits);
    }
};

fn decode(
    encoding_table: []const Encoding,
    decoding_table: []const Decoding,
    reader: anytype,
    writer: anytype,
    num_bits: u32,
) !void {
    // Initialize buffer with input data.
    var bit_buffer = BitBuffer{};
    while (bit_buffer.num_bits < decoding_bits) {
        const byte = try reader.readByte();
        try bit_buffer.addBits(byte, 8);
    }
    // Decode (most) input data. Some data may still be in the buffer after
    // this loop.
    while (bit_buffer.num_bits >= decoding_bits) {
        const decoding_idx = try bit_buffer.peekBits(u16, decoding_bits);
        const decoding = decoding_table[decoding_idx];
        switch (decoding) {
            .short => {
                bit_buffer.num_bits -= decoding.short.len;
                try writer.writeInt(
                    u16,
                    decoding.short.val,
                    std.builtin.Endian.little,
                );
            },
            .long => {
                // For long codes, we have to search the list stored in the
                // decoding table for a matching entry.
                var found = false;
                for (decoding.long.items) |encoding_idx| {
                    const encoding = encoding_table[encoding_idx];
                    const code = try bit_buffer.peekBits(u16, encoding.len);
                    if (encoding.val == code) {
                        bit_buffer.num_bits -= encoding.len;
                        try writer.writeInt(
                            u16,
                            decoding.short.val,
                            std.builtin.Endian.little,
                        );
                        found = true;
                        break;
                    }
                }
                if (!found) return error.InvalidCode;
            },
            .empty => return error.InvalidCode,
        }
        // Refill buffer if necessary.
        while (bit_buffer.num_bits < decoding_bits) {
            const byte = reader.readByte() catch |err| {
                if (err != error.EndOfStream) return err;
                break;
            };
            try bit_buffer.addBits(byte, 8);
        }
    }
    std.debug.assert(bit_buffer.num_bits < decoding_bits);
    // The compressed data isn't always byte-aligned at the end, so we need to
    // discard the extra bits we read.
    const extra_bits = (8 - @as(i32, @intCast(num_bits))) & 7;
    try bit_buffer.dumpBits(@intCast(extra_bits));
    // Decode any data still in the buffer.
    while (bit_buffer.num_bits > 0) {
        const shift = decoding_bits - bit_buffer.num_bits;
        const decoding_idx = (bit_buffer.bits << shift) & decoding_mask;
        const decoding = decoding_table[decoding_idx];
        switch (decoding) {
            .short => {
                if (decoding.short.len > bit_buffer.num_bits) {
                    break;
                }
                bit_buffer.num_bits -= decoding.short.len;
                try writer.writeInt(
                    u16,
                    decoding.short.val,
                    std.builtin.Endian.little,
                );
            },
            else => return error.InvalidCode,
        }
    }
}

test "decode" {
    // Adapted from tests in the 'exrs' project: https://github.com/johannesvollmer/exrs
    const allocator = std.testing.allocator;
    const uncompressed = [_]u16{
        3852,  2432,  33635, 49381, 10100, 15095, 62693, 63738, 62359, 5013,
        7715,  59875, 28182, 34449, 19983, 20399, 63407, 29486, 4877,  26738,
        44815, 14042, 46091, 48228, 25682, 35412, 7582,  65069, 6632,  54124,
        13798, 27503, 52154, 61961, 30474, 46880, 39097, 15754, 52897, 42371,
        54053, 14178, 48276, 34591, 42602, 32126, 42062, 31474, 16274, 55991,
        2882,  17039, 56389, 20835, 57057, 54081, 3414,  33957, 52584, 10222,
        25139, 40002, 44980, 1602,  48021, 19703, 6562,  61777, 41582, 201,
        31253, 51790, 15888, 40921, 3627,  12184, 16036, 26349, 3159,  29002,
        14535, 50632, 18118, 33583, 18878, 59470, 32835, 9347,  16991, 21303,
        26263, 8312,  14017, 41777, 43240, 3500,  60250, 52437, 45715, 61520,
    };
    const compressed = [_]u8{
        0x10, 0x9,  0xb4, 0xe4, 0x4c, 0xf7, 0xef, 0x42, 0x87, 0x6a, 0xb5, 0xc2,
        0x34, 0x9e, 0x2f, 0x12, 0xae, 0x21, 0x68, 0xf2, 0xa8, 0x74, 0x37, 0xe1,
        0x98, 0x14, 0x59, 0x57, 0x2c, 0x24, 0x3b, 0x35, 0x6c, 0x1b, 0x8b, 0xcc,
        0xe6, 0x13, 0x38, 0xc,  0x8e, 0xe2, 0xc,  0xfe, 0x49, 0x73, 0xbc, 0x2b,
        0x7b, 0x9,  0x27, 0x79, 0x14, 0xc,  0x94, 0x42, 0xf8, 0x7c, 0x1,  0x8d,
        0x26, 0xde, 0x87, 0x26, 0x71, 0x50, 0x45, 0xc6, 0x28, 0x40, 0xd5, 0xe,
        0x8d, 0x8,  0x1e, 0x4c, 0xa4, 0x79, 0x57, 0xf0, 0xc3, 0x6d, 0x5c, 0x6d,
        0xc0,
    };
    const num_bits = 674;
    // Build encoding and decoding tables.
    const frequencies = try countFrequencies(&uncompressed, allocator);
    defer allocator.free(frequencies);
    const result = try buildEncodingTable(frequencies, allocator);
    defer allocator.free(result.encoding_table);
    const encoding_table = result.encoding_table;
    const min_code_idx = result.min_code_idx;
    const max_code_idx = result.max_code_idx;
    buildCanonicalEncodings(encoding_table);
    const decoding_table = try buildDecodingTable(
        encoding_table,
        min_code_idx,
        max_code_idx,
        allocator,
    );
    defer allocator.free(decoding_table);
    // Decode compressed data
    var stream = std.io.fixedBufferStream(&compressed);
    var reader = stream.reader();
    var decoded = std.ArrayList(u8).init(allocator);
    defer decoded.deinit();
    var writer = decoded.writer();
    try decode(encoding_table, decoding_table, &reader, &writer, num_bits);
    const output = @as([]u16, @alignCast(std.mem.bytesAsSlice(u16, decoded.items)));
    try expect(std.mem.eql(u16, &uncompressed, output));
}

test "encode then decode" {
    // Adapted from tests in the 'exrs' project: https://github.com/johannesvollmer/exrs
    const allocator = std.testing.allocator;
    const uncompressed = [_]u16{
        3852,  2432,  33635, 49381, 10100, 15095, 62693, 63738, 62359, 5013,
        7715,  59875, 28182, 34449, 19983, 20399, 63407, 29486, 4877,  26738,
        44815, 14042, 46091, 48228, 25682, 35412, 7582,  65069, 6632,  54124,
        13798, 27503, 52154, 61961, 30474, 46880, 39097, 15754, 52897, 42371,
        54053, 14178, 48276, 34591, 42602, 32126, 42062, 31474, 16274, 55991,
        2882,  17039, 56389, 20835, 57057, 54081, 3414,  33957, 52584, 10222,
        25139, 40002, 44980, 1602,  48021, 19703, 6562,  61777, 41582, 201,
        31253, 51790, 15888, 40921, 3627,  12184, 16036, 26349, 3159,  29002,
        14535, 50632, 18118, 33583, 18878, 59470, 32835, 9347,  16991, 21303,
        26263, 8312,  14017, 41777, 43240, 3500,  60250, 52437, 45715, 61520,
    };
    // Build encoding table for data.
    const frequencies = try countFrequencies(&uncompressed, allocator);
    defer allocator.free(frequencies);
    const result = try buildEncodingTable(frequencies, allocator);
    const encoding_table = result.encoding_table;
    const min_code_idx = result.min_code_idx;
    const max_code_idx = result.max_code_idx;
    defer allocator.free(encoding_table);
    buildCanonicalEncodings(encoding_table);
    // Encode data into output buffer.
    var encoded = std.ArrayList(u8).init(allocator);
    defer encoded.deinit();
    var writer = encoded.writer();
    const num_bits = try encode(&uncompressed, encoding_table, max_code_idx, writer);
    // Build encoding and decoding tables.
    buildCanonicalEncodings(encoding_table);
    const decoding_table = try buildDecodingTable(
        encoding_table,
        min_code_idx,
        max_code_idx,
        allocator,
    );
    defer allocator.free(decoding_table);
    // Decode compressed data
    var stream = std.io.fixedBufferStream(encoded.items);
    var reader = stream.reader();
    var decoded = std.ArrayList(u8).init(allocator);
    defer decoded.deinit();
    writer = decoded.writer();
    try decode(encoding_table, decoding_table, &reader, &writer, num_bits);
    const output = @as([]u16, @alignCast(std.mem.bytesAsSlice(u16, decoded.items)));
    try expect(std.mem.eql(u16, &uncompressed, output));
}

const short_zerocode_run = 59;
const long_zerocode_run = 63;
const shortest_long_run = 2 + long_zerocode_run - short_zerocode_run;
const longest_long_run = 255 + shortest_long_run;

fn packEncodingTable(
    encoding_table: []Encoding,
    min_code_idx: u32,
    max_code_idx: u32,
    writer: anytype,
) !u32 {
    var num_bits: u32 = 0;
    var bit_writer = std.io.bitWriter(std.builtin.Endian.big, writer);
    var run_len: u8 = 0;
    for (min_code_idx..max_code_idx + 1) |encoding_idx| {
        const encoding = encoding_table[encoding_idx];
        if ((encoding.len == 0) and (run_len < longest_long_run)) {
            run_len += 1;
            continue;
        }
        if (run_len >= shortest_long_run) {
            try bit_writer.writeBits(@as(u6, long_zerocode_run), 6);
            try bit_writer.writeBits(run_len - shortest_long_run, 8);
            num_bits += 14;
        } else if (run_len >= 2) {
            try bit_writer.writeBits(run_len - 2 + short_zerocode_run, 6);
            num_bits += 6;
        } else if (run_len == 1) {
            try bit_writer.writeBits(@as(u6, 0), 6);
            num_bits += 6;
        }
        run_len = 0;
        try bit_writer.writeBits(encoding.len, 6);
        num_bits += 6;
    }
    try bit_writer.flushBits();
    return (num_bits + 7) / 8;
}

fn readEncodingTable(
    reader: anytype,
    min_code_idx: u32,
    max_code_idx: u32,
    allocator: std.mem.Allocator,
) ![]Encoding {
    const encoding_table = try allocator.alloc(Encoding, encoding_table_size);
    @memset(encoding_table, .{ .val = 0, .len = 0 });
    var bit_reader = std.io.bitReader(std.builtin.Endian.big, reader);
    var encoding_idx = min_code_idx;
    while (encoding_idx <= max_code_idx) {
        const encoding_len = try bit_reader.readBitsNoEof(u6, 6);
        if (encoding_len == long_zerocode_run) {
            const run_len = try bit_reader.readBitsNoEof(u8, 8) + shortest_long_run;
            if (encoding_idx + run_len > max_code_idx + 1) {
                return error.EncodingTableTooLong;
            }
            encoding_idx += run_len;
            continue;
        }
        if (encoding_len >= short_zerocode_run) {
            const run_len = encoding_len - short_zerocode_run + 2;
            if (encoding_idx + run_len > max_code_idx + 1) {
                return error.EncodingTableTooLong;
            }
            encoding_idx += run_len;
            continue;
        }
        encoding_table[encoding_idx].len = encoding_len;
        encoding_idx += 1;
    }
    return encoding_table;
}

test "pack and unpack encoding table" {
    const allocator = std.testing.allocator;
    const data = [_]u16{
        1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5,
        5, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    };
    // Build encoding table.
    const frequencies = try countFrequencies(&data, allocator);
    defer allocator.free(frequencies);
    const result = try buildEncodingTable(frequencies, allocator);
    defer allocator.free(result.encoding_table);
    const encoding_table = result.encoding_table;
    const min_code_idx = result.min_code_idx;
    const max_code_idx = result.max_code_idx;
    buildCanonicalEncodings(encoding_table);
    // Pack encoding table.
    var packed_table = std.ArrayList(u8).init(allocator);
    defer packed_table.deinit();
    const writer = packed_table.writer();
    _ = try packEncodingTable(
        encoding_table,
        min_code_idx,
        max_code_idx,
        writer,
    );
    // Unpack encoding table.
    var stream = std.io.fixedBufferStream(packed_table.items);
    const reader = stream.reader();
    const unpacked = try readEncodingTable(
        reader,
        min_code_idx,
        max_code_idx,
        allocator,
    );
    defer allocator.free(unpacked);
    buildCanonicalEncodings(unpacked);
    // The encoding tables should be identical.
    for (encoding_table, unpacked) |e1, e2| {
        try expect(std.meta.eql(e1, e2));
    }
}

const EncodingMetadata = packed struct {
    min_code_idx: u32,
    max_code_idx: u32,
    table_size: u32,
    num_bits: u32,
    padding: u32 = 0,
};

pub fn compress(
    src: []const u16,
    dst: *std.io.StreamSource,
    allocator: std.mem.Allocator,
) !void {
    const frequencies = try countFrequencies(src, allocator);
    defer allocator.free(frequencies);
    const result = try buildEncodingTable(frequencies, allocator);
    defer allocator.free(result.encoding_table);
    const encoding_table = result.encoding_table;
    const min_code_idx = result.min_code_idx;
    const max_code_idx = result.max_code_idx;
    buildCanonicalEncodings(encoding_table);

    const metadata_pos = try dst.getPos();
    try dst.seekBy(5 * @sizeOf(u32));
    const table_size = try packEncodingTable(
        encoding_table,
        min_code_idx,
        max_code_idx,
        dst.writer(),
    );
    const num_bits = try encode(
        src,
        encoding_table,
        max_code_idx,
        dst.writer(),
    );
    try dst.seekTo(metadata_pos);
    try dst.writer().writeInt(u32, min_code_idx, std.builtin.Endian.little);
    try dst.writer().writeInt(u32, max_code_idx, std.builtin.Endian.little);
    try dst.writer().writeInt(u32, table_size, std.builtin.Endian.little);
    try dst.writer().writeInt(u32, num_bits, std.builtin.Endian.little);
    try dst.writer().writeInt(u32, 0, std.builtin.Endian.little);
}

pub fn decompress(
    reader: anytype,
    writer: anytype,
    allocator: std.mem.Allocator,
) !void {
    const min_code_idx = try reader.readInt(u32, std.builtin.Endian.little);
    const max_code_idx = try reader.readInt(u32, std.builtin.Endian.little);
    const table_size = try reader.readInt(u32, std.builtin.Endian.little);
    const num_bits = try reader.readInt(u32, std.builtin.Endian.little);
    const padding = try reader.readInt(u32, std.builtin.Endian.little);
    _ = padding;
    _ = table_size;
    const unpacked = try readEncodingTable(
        reader,
        min_code_idx,
        max_code_idx,
        allocator,
    );
    defer allocator.free(unpacked);
    buildCanonicalEncodings(unpacked);
    const decoding_table = try buildDecodingTable(
        unpacked,
        min_code_idx,
        max_code_idx,
        allocator,
    );
    defer allocator.free(decoding_table);
    const num_bytes = ((num_bits + 7) & 0xFFFFFFF8) >> 3;
    var limited_reader = std.io.limitedReader(reader, num_bytes);
    try decode(unpacked, decoding_table, limited_reader.reader(), writer, num_bits);
}

test "compress then decompress" {
    const allocator = std.testing.allocator;
    const src = [_]u16{
        1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5,
        5, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    };
    const dst_buffer = try allocator.alloc(u8, 256);
    defer allocator.free(dst_buffer);
    @memset(dst_buffer, 0);
    var dst = std.io.StreamSource{
        .buffer = std.io.fixedBufferStream(dst_buffer),
    };
    try compress(&src, &dst, allocator);

    try dst.seekTo(0);

    const decoded_buffer = try allocator.alloc(u8, 256);
    defer allocator.free(decoded_buffer);
    @memset(decoded_buffer, 0);
    var decoded = std.io.StreamSource{
        .buffer = std.io.fixedBufferStream(decoded_buffer),
    };
    try decompress(
        dst.reader(),
        decoded.writer(),
        allocator,
    );
    const output = @as([]u16, @alignCast(std.mem.bytesAsSlice(u16, decoded_buffer)));
    try expect(std.mem.eql(u16, output[0..src.len], &src));
}
