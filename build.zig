const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Add run executable
    const run_exe = b.addExecutable(.{
        .name = "run", 
        .root_source_file = .{ .path = "run.c" },
        .target = target,
        // .link_libc = true,
        .optimize = optimize,
    });
    // run_exe.linkSystemLibrary("m");

    // Add runq executable 
    const runq_exe = b.addExecutable(.{
        .name = "runq",
        .root_source_file = .{ .path = "runq.c" },
        .target = target,
        // .link_libc = true,
        .optimize = optimize,
    });
    // runq_exe.linkSystemLibrary("m");

    // Add test executable
    const test_exe = b.addExecutable(.{
        .name = "test",
        .root_source_file = .{ .path = "test.c" }, 
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(run_exe);
    b.installArtifact(runq_exe);
    b.installArtifact(test_exe);
}