const std = @import("std");
const testing = std.testing;

pub const piz = @import("piz/lut.zig");

test {
    std.testing.refAllDecls(@This());
}
