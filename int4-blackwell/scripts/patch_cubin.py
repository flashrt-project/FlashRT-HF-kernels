#!/usr/bin/env python3
"""Flip SASS bits 78/79 of every OMMA.SF instruction in a cubin.

Claim under test (RTX 5090 / SM120): in the 128-bit encoding of
OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X, bit 78 selects the A-operand format
and bit 79 the B-operand format; 0 = E2M1, 1 = E0M3 (INT4, codebook -7..7).

Instructions are 16 bytes little-endian, so bit 78 = byte 9 bit 6 and
bit 79 = byte 9 bit 7 within each instruction word.

Usage: patch_cubin.py <in.cubin> <out.cubin> <a|b|ab>
"""
import re
import subprocess
import sys

NVDISASM = "/usr/local/cuda/bin/nvdisasm"
CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
READELF = "readelf"


def find_omma_sites(cubin):
    """Return list of (function, instr_offset) for OMMA.SF instructions."""
    sass = subprocess.run([CUOBJDUMP, "-sass", cubin], capture_output=True,
                          text=True, check=True).stdout
    sites, fn = [], None
    for line in sass.splitlines():
        m = re.search(r"Function\s*:\s*(\S+)", line)
        if m:
            fn = m.group(1)
            continue
        m = re.match(r"\s*/\*([0-9a-fA-F]+)\*/\s+(.*)", line)
        if m and "OMMA" in m.group(2) and ".SF." in m.group(2):
            sites.append((fn, int(m.group(1), 16), m.group(2).split(";")[0].strip()))
    return sites


def section_offsets(cubin):
    """Map .text.<fn> section name -> file offset via readelf."""
    out = subprocess.run([READELF, "-S", "-W", cubin], capture_output=True,
                         text=True, check=True).stdout
    offs = {}
    for line in out.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\.text\.\S+)\s+\S+\s+\S+\s+([0-9a-fA-F]+)", line)
        if m:
            offs[m.group(1)] = int(m.group(2), 16)
    return offs


def main():
    src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    assert mode in ("a", "b", "ab")
    mask = (0x40 if "a" in mode else 0) | (0x80 if "b" in mode else 0)

    data = bytearray(open(src, "rb").read())
    sites = find_omma_sites(src)
    secs = section_offsets(src)
    if not sites:
        sys.exit("no OMMA.SF instructions found")

    n = 0
    for fn, ioff, text in sites:
        sec = f".text.{fn}"
        if sec not in secs:
            sys.exit(f"section {sec} not found in ELF")
        foff = secs[sec] + ioff + 9  # byte 9 of the 16-byte instruction
        old = data[foff]
        data[foff] = old | mask
        n += 1
        print(f"patched {fn}+0x{ioff:x}: byte9 0x{old:02x} -> 0x{data[foff]:02x}  ({text[:60]})")

    open(dst, "wb").write(data)
    print(f"wrote {dst} ({n} instructions patched, mode={mode})")


if __name__ == "__main__":
    main()
