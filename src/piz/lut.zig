//! This module provides functions for compressing the range of a dataset
//! using bitmaps and lookup tables.

const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;

const u16_range = std.math.maxInt(u16) + 1;
const bitmap_size = u16_range >> 3;

const BitmapResult = struct {
    bitmap: []u8,
    min_non_zero: u16,
    max_non_zero: u16,
};

fn generateBitmap(
    data: []const u16,
    allocator: std.mem.Allocator,
) !BitmapResult {
    const bitmap = try allocator.alloc(u8, bitmap_size);
    @memset(bitmap, 0);
    // array index = upper 13 bits
    //   bit index = lower  3 bits
    for (data) |x| {
        bitmap[x >> 3] |= (@as(u8, 1) << @as(u3, @intCast(x & 7)));
    }
    // don't store zero, its existence is assumed
    bitmap[0] &= 0b11111110;
    // compute the first and last non zero bytes
    var min_non_zero: u16 = bitmap_size - 1;
    var max_non_zero: u16 = 0;
    for (0..bitmap_size) |i| {
        const j: u16 = @intCast(i);
        if (bitmap[j] == 0) continue;
        min_non_zero = @min(j, min_non_zero);
        max_non_zero = @max(j, max_non_zero);
    }
    assert(min_non_zero <= max_non_zero);
    return .{
        .bitmap = bitmap,
        .min_non_zero = min_non_zero,
        .max_non_zero = max_non_zero,
    };
}

test "bitmap basic functionality" {
    const allocator = std.testing.allocator;
    const data = [_]u16{ 3, 10, 5, 1 };
    const result = try generateBitmap(&data, allocator);
    defer allocator.free(result.bitmap);
    try expect(result.min_non_zero == 0);
    try expect(result.max_non_zero == 1);
    //  1 -> byte 0, bit 1 (00000010)
    //  3 -> byte 0, bit 3 (00001000)
    //  5 -> byte 0, bit 5 (00100000)
    // 10 -> byte 1, bit 2 (00000100)
    try expect(result.bitmap[0] == 0b00101010);
    try expect(result.bitmap[1] == 0b00000100);
    for (result.bitmap[2..]) |byte| {
        try expect(byte == 0);
    }
}

test "bitmap largest value" {
    const allocator = std.testing.allocator;
    const data = [_]u16{u16_range - 1};
    const result = try generateBitmap(&data, allocator);
    defer allocator.free(result.bitmap);
    try expect(result.min_non_zero == bitmap_size - 1);
    try expect(result.max_non_zero == bitmap_size - 1);
    try expect(result.bitmap[bitmap_size - 1] == 0b10000000);
    for (result.bitmap[0 .. bitmap_size - 1]) |byte| {
        try expect(byte == 0);
    }
}

const LUTResult = struct {
    lut: []u16,
    num_nonzero_values: u16,
};

fn generateForwardLUT(
    bitmap: []const u8,
    allocator: std.mem.Allocator,
) !LUTResult {
    // Map values of set bits to sequential numbers.
    const lut = try allocator.alloc(u16, u16_range);
    @memset(lut, 0);
    var k: u16 = 0;
    for (0..u16_range) |i| {
        const mask = @as(u8, 1) << @as(u3, @intCast(i & 7));
        const bit_is_set = (bitmap[i >> 3] & mask) != 0;
        if ((i == 0) or bit_is_set) {
            lut[i] = k;
            k += 1;
        }
    }
    return .{ .lut = lut, .num_nonzero_values = k - 1 };
}

test "forward LUT" {
    const allocator = std.testing.allocator;
    const data = [_]u16{ 3, 10, 5, 1 };
    const bitmap = (try generateBitmap(&data, allocator)).bitmap;
    defer allocator.free(bitmap);
    const result = try generateForwardLUT(bitmap, allocator);
    defer allocator.free(result.lut);
    try expect(result.num_nonzero_values == 4);
    try expect(result.lut[0] == 0);
    try expect(result.lut[1] == 1);
    try expect(result.lut[3] == 2);
    try expect(result.lut[5] == 3);
    try expect(result.lut[10] == 4);
}

fn generateReverseLUT(
    bitmap: []const u8,
    allocator: std.mem.Allocator,
) !LUTResult {
    // Map sequential numbers to values of set bits.
    const lut = try allocator.alloc(u16, u16_range);
    @memset(lut, 0);
    var k: u16 = 0;
    for (0..u16_range) |i| {
        const mask = @as(u8, 1) << @as(u3, @intCast(i & 7));
        const bit_is_set = (bitmap[i >> 3] & mask) != 0;
        if ((i == 0) or bit_is_set) {
            lut[k] = @intCast(i);
            k += 1;
        }
    }
    return .{ .lut = lut, .num_nonzero_values = k - 1 };
}

test "reverse LUT" {
    const allocator = std.testing.allocator;
    const data = [_]u16{ 3, 10, 5, 1 };
    const bitmap = (try generateBitmap(&data, allocator)).bitmap;
    defer allocator.free(bitmap);
    const result = try generateReverseLUT(bitmap, allocator);
    defer allocator.free(result.lut);
    try expect(result.num_nonzero_values == 4);
    try expect(result.lut[0] == 0);
    try expect(result.lut[1] == 1);
    try expect(result.lut[2] == 3);
    try expect(result.lut[3] == 5);
    try expect(result.lut[4] == 10);
}

fn applyLUT(lut: []const u16, data: []u16) void {
    for (data) |*x| {
        x.* = lut[x.*];
    }
}

test "apply forward LUT" {
    const allocator = std.testing.allocator;
    var data = [_]u16{ 3, 10, 5, 1 };
    const bitmap = (try generateBitmap(&data, allocator)).bitmap;
    defer allocator.free(bitmap);
    const lut = (try generateForwardLUT(bitmap, allocator)).lut;
    defer allocator.free(lut);
    applyLUT(lut, &data);
    try expect(data[0] == 2);
    try expect(data[1] == 4);
    try expect(data[2] == 3);
    try expect(data[3] == 1);
}

test "apply LUTs roundtrip" {
    const allocator = std.testing.allocator;
    var data = [_]u16{ 3, 10, 5, 1 };
    const bitmap = (try generateBitmap(&data, allocator)).bitmap;
    defer allocator.free(bitmap);
    const forward_lut = (try generateForwardLUT(bitmap, allocator)).lut;
    defer allocator.free(forward_lut);
    const reverse_lut = (try generateReverseLUT(bitmap, allocator)).lut;
    defer allocator.free(reverse_lut);
    applyLUT(forward_lut, &data);
    applyLUT(reverse_lut, &data);
    try expect(data[0] == 3);
    try expect(data[1] == 10);
    try expect(data[2] == 5);
    try expect(data[3] == 1);
}
