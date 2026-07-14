#!/usr/bin/env bash
# Local cluster defaults for this checkout. Export your own values before invoking
# the pipeline to override any of these.

_site_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_site_project_root="$(cd "${_site_env_dir}/.." && pwd)"
_site_rosetta_bin="${_site_project_root}/scratch/rosetta_src_2017.52.59948_bundle/main/source/build/src/release/linux/4.18/64/x86/gcc/9.4/default/rosetta_scripts.default.linuxgccrelease"

if [ -x "${_site_rosetta_bin}" ]; then
  export ROSETTA_SCRIPTS_BIN="${ROSETTA_SCRIPTS_BIN:-${_site_rosetta_bin}}"
fi

unset _site_env_dir _site_project_root _site_rosetta_bin