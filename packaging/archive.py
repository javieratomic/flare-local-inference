# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package a native Flare executable and its license notices as tgz or zip."""

import argparse
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

from licenses import ROOT, TARGETS, collect, toolchain_notices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error("--binary must identify the executable built with the current Cargo lockfile and rustc")
    if args.output.exists():
        parser.error("--output already exists")
    if not str(args.output).endswith((".zip", ".tgz", ".tar.gz")):
        parser.error("--output must end in .zip, .tgz, or .tar.gz")
    with tempfile.TemporaryDirectory(prefix="flare-release-") as directory:
        stage = Path(directory) / "flare"
        stage.mkdir()
        executable = "flare.exe" if "windows" in args.target else "flare"
        shutil.copy2(args.binary, stage / executable)
        for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
            shutil.copyfile(ROOT / name, stage / name)
        shutil.copytree(ROOT / "helios/licenses", stage / "licenses")
        collect(stage / "third_party/rust", [args.target])
        toolchain_notices(stage / "third_party/rust-standard-library")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".zip":
            with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(stage.rglob("*")):
                    if file.is_file():
                        archive.write(file, file.relative_to(stage.parent))
        else:
            with tarfile.open(args.output, "w:gz") as archive:
                archive.add(stage, arcname="flare")
    print(args.output)


if __name__ == "__main__":
    main()
