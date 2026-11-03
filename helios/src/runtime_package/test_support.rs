/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixture publisher shared by native package tests; never an official release key.

use super::{
    AuthorizedPackage, Publisher, VerifiedPackage, payload::Files, protection::AuthorizedKind,
};
use crate::Result;
use ring::signature::{Ed25519KeyPair, KeyPair};
use std::{io::Read, path::Path};

pub fn publisher() -> Publisher {
    let key = Ed25519KeyPair::from_seed_unchecked(&[42; 32]).unwrap();
    Publisher {
        development: false,
        name: "Flare test publisher".into(),
        namespace: "flare-tests".into(),
        public_key: key.public_key().as_ref().try_into().unwrap(),
    }
}
pub fn authorize(path: impl AsRef<Path>) -> AuthorizedPackage {
    VerifiedPackage::open(path.as_ref(), &[publisher()], None)
        .unwrap()
        .authorize("flare-tests", false)
        .unwrap()
}
/// Authenticate a package, every rank or only `rank` of a multi-host package, and digest the
/// plaintext of each verified rank's executables and supporting resources and of the assets.
pub fn inspect(
    path: &Path,
    namespace: &str,
    verify_weights: bool,
    publishers: &[Publisher],
    rank: Option<usize>,
) -> Result<Vec<(u64, String)>> {
    let package =
        VerifiedPackage::open(path, publishers, rank)?.authorize(namespace, verify_weights)?;
    if let AuthorizedKind::Payload(payload) = &package.kind {
        // Payload packages stream every authenticated resource: executables are decompressed per
        // chunk and plaintext resources are read at their pinned ranges.
        let source = payload.source();
        let mut contents = Vec::new();
        let mut ordinal = 0u64;
        for package in payload.rank_packages() {
            for key in package.resources.keys() {
                // Digest from the streaming reader: executables decompress authenticated chunks and stored
                // payloads read their pinned range, so no resource is materialized whole.
                let mut reader = Files::resource_reader(source, package, key)?;
                let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
                let mut buffer = vec![0u8; 65536];
                loop {
                    let count = reader.read(&mut buffer).map_err(|e| e.to_string())?;
                    if count == 0 {
                        break;
                    }
                    digest.update(&buffer[..count]);
                }
                contents.push((ordinal, super::protection::hex(digest.finish().as_ref())));
                ordinal += 1;
            }
        }
        return Ok(contents);
    }
    let mut contents = Vec::new();
    let mut spans = Vec::new();
    for rank in &package
        .legacy_index()
        .ok_or_else(|| crate::runtime_package::format::invalid("legacy package required"))?
        .ranks
    {
        for name in rank
            .modules
            .iter()
            .map(|m| &m.file)
            .chain(rank.graphs.iter().map(|g| &g.resource))
            .chain(
                rank.modules
                    .iter()
                    .flat_map(|m| m.globals.iter().map(|g| &g.file)),
            )
        {
            spans.push(&rank.resources[name]);
        }
    }
    spans.extend(
        package
            .legacy_index()
            .ok_or_else(|| crate::runtime_package::format::invalid("legacy package required"))?
            .assets
            .iter()
            .filter_map(|asset| asset.payload.as_ref()),
    );
    for span in spans {
        let (source, offset) = package
            .legacy_archive()
            .ok_or_else(|| crate::runtime_package::format::invalid("legacy package required"))?
            .source(span)?;
        let mut reader = package
            .legacy_archive()
            .ok_or_else(|| crate::runtime_package::format::invalid("legacy package required"))?
            .protection
            .reader(source.file.clone(), offset, span)?;
        let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
        let mut buffer = vec![0; 65536];
        loop {
            let count = reader.read(&mut buffer).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        contents.push((
            span.offset,
            super::protection::hex(digest.finish().as_ref()),
        ));
    }
    Ok(contents)
}

pub fn inspect_graph(path: &Path, name: &str) -> Result<()> {
    let package =
        VerifiedPackage::open(path, &[publisher()], None)?.authorize("flare-tests", false)?;
    if let AuthorizedKind::Payload(payload) = &package.kind {
        // Same record framing as the legacy graph bodies, read through the streaming source.
        let source = payload.source();
        let package = payload
            .rank_packages()
            .first()
            .ok_or_else(|| crate::runtime_package::format::invalid("package holds no rank"))?;
        let graph = package
            .graphs
            .iter()
            .find(|graph| graph.name == name)
            .ok_or("missing graph")?;
        let mut records = Files::records(source, package, &graph.resource)?;
        for _ in 0..graph.node_count {
            records.read::<super::protobuf::wire::Node>()?;
        }
        for _ in 0..graph.edge_count {
            records.read::<super::protobuf::wire::Edge>()?;
        }
        return records.finish();
    }
    let mut index = package
        .legacy_index()
        .ok_or_else(|| crate::runtime_package::format::invalid("legacy package required"))?
        .clone();
    let bundle = super::protobuf::decode(index.ranks.remove(0))?;
    let graph = bundle
        .graphs
        .iter()
        .find(|g| g.name == name)
        .ok_or("missing graph")?;
    let mut records = package
        .legacy_archive()
        .ok_or_else(|| crate::runtime_package::format::invalid("legacy package required"))?
        .records(&bundle, &graph.resource)?;
    for _ in 0..graph.node_count {
        records.read::<super::protobuf::wire::Node>()?;
    }
    for _ in 0..graph.edge_count {
        records.read::<super::protobuf::wire::Edge>()?;
    }
    records.finish()
}
