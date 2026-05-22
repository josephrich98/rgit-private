#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import os


def find_nifti_files(input_dir):
    exts = [".nii", ".nii.gz", ".mgz"]
    files = []

    for ext in exts:
        files.extend(Path(input_dir).rglob(f"*{ext}"))

    return sorted(files)


def run_fastsurfer(
    input_file,
    output_dir,
    license_file=None,
    threads=8,
    seg_only=True,
):
    input_file = Path(input_file).resolve()
    output_dir = Path(output_dir).resolve()

    subject_id = input_file.name
    subject_id = subject_id.replace(".nii.gz", "")
    subject_id = subject_id.replace(".nii", "")
    subject_id = subject_id.replace(".mgz", "")

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{input_file.parent}:/data",
        "-v",
        f"{output_dir}:/output",
    ]

    # Optional FreeSurfer license mount
    if license_file is not None:
        license_file = Path(license_file).resolve()

        docker_cmd.extend([
            "-v",
            f"{license_file.parent}:/fs_license",
        ])

    docker_cmd.append("deepmi/fastsurfer:latest")

    docker_cmd.extend([
        "--t1",
        f"/data/{input_file.name}",
        "--sid",
        subject_id,
        "--sd",
        "/output",
        "--threads",
        str(threads),
        "--device",
        "cuda",
    ])

    if seg_only:
        docker_cmd.append("--seg_only")

    if license_file is not None:
        docker_cmd.extend([
            "--fs_license",
            f"/fs_license/{license_file.name}"
        ])

    print("\nRunning:")
    print(" ".join(docker_cmd))
    print()

    subprocess.run(docker_cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Directory containing MRI files",
    )

    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "-l", "--license_file",
        default=os.getenv("FREESURFER_HOME") + "/license.txt",
        help="FreeSurfer license file (optional if using --seg_only)",
    )

    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "-f", "--full",
        action="store_true",
        help="Run full FastSurfer pipeline instead of seg_only",
    )

    args = parser.parse_args()

    input_files = find_nifti_files(args.input_dir)

    if not os.path.exists(args.license_file) and args.full:
        raise FileNotFoundError(f"License file not found: {args.license_file}")

    print(f"Found {len(input_files)} MRI files")

    for input_file in input_files:
        print(f"\nProcessing: {input_file}")

        run_fastsurfer(
            input_file=input_file,
            output_dir=args.output_dir,
            license_file=args.license_file,
            threads=args.threads,
            seg_only=not args.full,
        )


if __name__ == "__main__":
    main()