use std::env;
use std::path::PathBuf;

fn generate_bindings() {
    let crate_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR env var is not defined");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR env var is not defined"));
    let package_name = env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME env var is not defined");

    // Also generate the header file in the project root for easier access
    let root_header = PathBuf::from(&crate_dir).join("gosilero.h");

    // Create a default config
    let mut config = cbindgen::Config::default();
    config.language = cbindgen::Language::C;
    config.include_guard = Some("GOSILERO_H".to_string());
    config.includes = vec![
        "stdlib.h".to_string(),
        "stdint.h".to_string(),
        "stdbool.h".to_string(),
    ];
    config.header = Some("/* Auto-generated with cbindgen */".to_string());

    // Generate the bindings
    let bindings =
        cbindgen::generate_with_config(&crate_dir, config).expect("Unable to generate bindings");

    // Write the bindings to the output directory
    bindings.write_to_file(out_dir.join(format!("{}.h", package_name)));

    // Also write to the project root directory for easier access
    bindings.write_to_file(root_header);

    println!("cargo:rerun-if-changed=src/lib.rs");
}

fn main() {
    generate_bindings();
}
