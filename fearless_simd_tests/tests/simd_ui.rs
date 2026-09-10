// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(missing_docs, reason = "this integration test is not public API")]

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
#[ignore = "run explicitly in the native Linux CI job"]
fn simd_ui() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui/simd/pass/*.rs");
    tests.compile_fail("tests/ui/simd/fail/*.rs");

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let status = std::process::Command::new(env!("CARGO"))
        .args(["check", "--offline", "--manifest-path"])
        .arg(manifest_dir.join("tests/ui/simd/renamed/Cargo.toml"))
        .arg("--target-dir")
        .arg(manifest_dir.join("../target/tests/simd-renamed"))
        .status()
        .expect("renamed-dependency fixture should start cargo");
    assert!(
        status.success(),
        "renamed-dependency fixture should compile"
    );
}
