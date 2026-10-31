"""Size checks on a real GGUF: what setup.py says about a file of another length.

setup.py runs two checks on a model file, and either one stops the whole run:
  check_shards(s)          reads the header and asks whether every byte the tensor directory needs is in the file
  verify_sha256(s, ...)    compares the length with the published one, and hashes only when they agree

This tool makes one copy of your model file (in the same folder, not on the system drive), shortens the
copy by N bytes, and runs both checks on it, printing what each step did, what own_gguf_is_whole()
said, and what setup told the person.

The model file is only ever read: it is not renamed, not moved, not one byte written, so an interrupt,
an error or a power cut leaves it where it was, with the bytes it had. The cost is disk space, one copy
the size of the model -- keep that much free, or point --work-dir at another drive. The copy is deleted
when the run ends; one an earlier run could not delete is removed at the start of the next (only names
starting with `strata-drill-copy-` are touched, never your model file or its .done mark).

Usage, from the Strata folder:
    python tools/setup_size_check_drill.py --file <your.gguf>
(the copy goes beside the model file; --work-dir D:\\temp puts it elsewhere)

Without --file, a few hundred bytes of valid GGUF are built here and run through the same checks
(no network, no model file needed).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Every copy a run leaves behind starts with this; reserve that much space next to the model file
DRILL_PREFIX = "strata-drill-copy-"

ALIGN = 32          # GGUF tensors are aligned to 32 bytes
TAIL = 64           # how far back a rewrite reaches: two blocks, so an unaligned end is covered too
Q8_BYTES = 64       # one Q8_0 tensor (32 elements) is 64 bytes: a multiple of two blocks, so any 32-byte offset is aligned
HASH_NOTICE = 1 << 30   # past 1 GiB, warn first: the whole file is read (minutes on a big one)


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def tiny_gguf(tensors: int = 2, extra: int = 0) -> bytes:
    """A real GGUF v3: `tensors` Q8_0 tensors, then `extra` bytes at the end."""
    head = b"GGUF" + struct.pack("<IQQ", 3, tensors, 1)
    head += _s("general.architecture") + struct.pack("<I", 8) + _s("qwen4exp")
    for i in range(tensors):
        head += _s(f"blk.{i}.weight") + struct.pack("<I", 1) + struct.pack("<Q", 32)
        head += struct.pack("<IQ", 8, i * Q8_BYTES)
    head += b"\0" * (-len(head) % ALIGN)
    return head + b"\0" * (tensors * Q8_BYTES + extra)


def pinned(label: str | None, name: str) -> tuple[int, str] | None:
    """The file name's (published length, SHA-256) from setup.py's tables, when it is in one."""
    import setup

    tables = []
    for attr in sorted(dir(setup)):
        value = getattr(setup, attr)
        if attr.endswith("_FILES") and isinstance(value, dict) and name in value:
            entry = value[name]
            if isinstance(entry, tuple) and len(entry) >= 2 and isinstance(entry[0], int):
                tables.append((attr, entry[0], entry[1]))
    if not tables:
        return None
    if label:
        for attr, size, sha in tables:
            if attr.lower().startswith(label.lower().replace("-", "_")):
                return size, sha
        print(f"  [!]  --label {label} matches no table (there are: {', '.join(t[0] for t in tables)})", flush=True)
    elif len({t[1:] for t in tables}) > 1:
        print(f"  [!]  {name} has a different length in more than one table; using the first. Pick one with "
              f"--label: {', '.join(f'{t[0]}={t[1]:,}' for t in tables)}", flush=True)
    print(f"  Published length taken from setup.py's {tables[0][0]}", flush=True)
    return tables[0][1], tables[0][2]


def copy_name(name: str, length: int) -> str:
    """The copy's name: the model's name and its length, so a glance at the folder shows what it is."""
    return f"{DRILL_PREFIX}{Path(name).stem}-{length}.gguf"


def leftovers(folder: Path, name: str) -> list[Path]:
    """Copies an earlier run left in `folder` (matched by name, never by size)."""
    pattern = re.compile(re.escape(DRILL_PREFIX + Path(name).stem) + r"-\d+\.gguf$")
    try:
        return sorted(p for p in folder.iterdir() if p.is_file() and pattern.match(p.name))
    except OSError:
        return []


def free_bytes(folder: Path) -> int:
    try:
        return shutil.disk_usage(folder).free
    except OSError:
        return 1 << 62


def copied_to(source: Path, folder: Path, length: int) -> Path:
    """Copy the first `length` bytes of `source` into a new file in `folder`; `source` is only read.

    Not one byte more: extra bytes would blend into the padding at the end of the file and hide
    what really made it read as whole."""
    dest = folder / copy_name(source.name, length)
    total = source.stat().st_size
    if length > total:
        raise SystemExit(f"{source.name} holds {total:,} bytes, fewer than the {length:,} asked for: "
                         "a copy cannot be longer than the file it is made from")
    free = free_bytes(folder)
    if free < length:
        raise SystemExit(f"{folder} has only {free:,} bytes free and this run needs {length:,} (one copy): "
                         "free some space, or point --work-dir at a roomier drive")
    with source.open("rb") as fh:
        with open(dest, "wb") as out:
            left = length
            while left > 0:
                chunk = fh.read(min(left, 8 << 20))
                if not chunk:
                    raise SystemExit(f"{source} ended after {total - left:,} bytes")
                out.write(chunk)
                left -= len(chunk)
    return dest


#: One entry per case, printed again as the ✅ / ❌ summary at the end.
#: (number, what the case is, setup accepted it, setup's line when it refused, expected to accept)
RESULTS: list[tuple[int, str, bool, str, bool]] = []


def _run(call, quiet: bool, *args, **kwargs) -> tuple[bool, str, str]:
    """Run one of setup's checks with its output captured.

    Returns (passed, the [X] line when it refused, everything setup said)."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            call(*args, **kwargs)
    except SystemExit:
        text = buf.getvalue()
        if not quiet:
            print(text.rstrip(), flush=True)
        for line in text.splitlines():
            if "[X]" in line:
                return False, line.split("[X]", 1)[1].strip(), text.strip()
        return False, "(setup stopped without an [X] line)", text.strip()
    text = buf.getvalue().strip()
    if not quiet:
        print(text, flush=True)
    return True, "", text


def check(path: Path, size: int, sha: str, own: bool, setup, quiet: bool = True) -> tuple[bool, str, str]:
    """Run setup's two checks on `path`.

    Returns (accepted, setup's refusal line, everything setup said). The third value is kept
    verbatim: for a file that is used it is the "uses it as it is" warning, for one that is
    refused the [X] line and its hint.

    quiet: setup prints [X] and "Setup stopped" itself; that stays as evidence but is not scrolled
    past on the screen, since it is part of the drill rather than this run going wrong."""
    have = path.stat().st_size
    whole = setup.own_gguf_is_whole(path)
    print(f"  own_gguf_is_whole(): {have:,} bytes -> {'whole' if whole else 'not whole'}", flush=True)

    passed, reason, said = _run(setup.check_shards, quiet, [path])
    if not passed:
        return False, reason, said
    if have == size and size >= HASH_NOTICE:
        print(f"  Same length as the published file, so the SHA-256 is checked: reading {size / 1e9:.0f} GB, "
              f"{size / 2e9:.0f}-{size / 1e9:.0f} minutes", flush=True)
    elif have != size:
        print("  Another length, so no SHA-256 is computed (only whether it is a whole GGUF file)", flush=True)
    return _run(setup.verify_sha256, quiet, path, size, sha, own=own)


def go_back(source: Path, path: Path, length: int) -> None:
    """Make `path` hold the first `length` bytes of `source`, rewriting only the few blocks at its end.

    The window starts two alignment blocks back from the shorter of the two lengths, so it covers
    an unaligned end as well: a file of 58,487,015,904 bytes has no block boundary there."""
    tail = max(0, min(length, source.stat().st_size) - TAIL)
    with path.open("r+b") as fh:
        fh.truncate(length)
        fh.seek(tail)
        left = length - tail
        with source.open("rb") as src:
            src.seek(tail)
            while left > 0:
                chunk = src.read(min(left, 8 << 20))
                if not chunk:
                    raise SystemExit(f"{source} ended after {length - left:,} bytes")
                fh.write(chunk)
                left -= len(chunk)


def same_tail(source: Path, path: Path, start: int) -> bool:
    """Whether the two files agree from `start` on: a small tail, never the whole file."""
    with source.open("rb") as a, path.open("rb") as b:
        a.seek(start)
        b.seek(start)
        while True:
            x, y = a.read(8 << 20), b.read(8 << 20)
            if x != y:
                return False
            if not x:
                return True


def back_to(source: Path, path: Path, have: int) -> None:
    """Make `path` a full copy of `source` again (length `have`).

    The length alone is not enough: the last case rewrote the tail, so the length can match while
    the bytes do not. Both the comparison and the rewrite start at `have - TAIL`."""
    if path.stat().st_size == have and same_tail(source, path, max(0, have - TAIL)):
        return
    start = max(0, have - TAIL)
    with path.open("r+b") as fh:
        fh.truncate(have)
        fh.seek(start)
        with source.open("rb") as src:
            src.seek(start)
            left = have - start
            while left > 0:
                chunk = src.read(min(left, 8 << 20))
                if not chunk:
                    raise SystemExit(f"{source} ended after {have - left:,} bytes")
                fh.write(chunk)
                left -= len(chunk)


def verdict(index: int, accepted: bool, expected: bool) -> str:
    """One case's result: what setup did, and whether that is what this case expects."""
    mark = "✅" if accepted == expected else "❌"
    word = "as expected" if accepted == expected else "NOT as expected"
    return f"  {mark} Case {index} {word}: setup {'accepted' if accepted else 'refused'}"


def run_cases(work: Path, data: Path, size: int, sha: str, cases, have: int) -> None:
    """Run every case on the copy `work`, then put its tail back the way it was.

    `data` is the model file, only read: it supplies the bytes written back and the comparison.
    Each case touches at most the last few blocks, so the copy ends up being a full copy again."""
    import setup

    tail = have
    try:
        for index, (part, length) in enumerate(cases, 1):
            if length != tail:
                go_back(data, work, length)
                tail = length
            print(f"\nCase {index}: {describe(length, have)}", flush=True)
            accepted, reason, said = check(work, size, sha, False, setup)
            # One thing decides what this case expects: whether the file is a whole GGUF one. If it is,
            # setup should accept it; if not, it should refuse. A length other than the published one is
            # not setup's business: a file that has been damaged enough to matter fails when it is loaded.
            expected = setup.own_gguf_is_whole(work)
            RESULTS.append((index, describe(length, have), accepted, reason, expected))

            if said:                                  # setup's own line, for a file it used and one it refused
                print(f"  setup said: {said}", flush=True)
            print(verdict(index, accepted, expected), flush=True)
            if accepted != expected:
                print(f"      this case expects setup to {'accept' if expected else 'refuse'}", flush=True)
    finally:
        if work.exists():
            back_to(data, work, have)      # tail and all, so the copy is whole again


def drill_real(source: Path, size: int, sha: str, cases, folder: Path) -> int:
    """A real model file: read, never written. One copy is made in `folder`, every case reshapes that
    copy, and the copy is deleted at the end.

    The model file is only ever read: not renamed, not moved, not one byte written. An interrupt, an
    error or a power cut leaves it where it was, with the bytes it had. The cost is disk space, one
    copy the size of the model (--work-dir puts it on another drive)."""
    have = source.stat().st_size
    print(f"Model file {source}\n  {have:,} bytes, published {size:,}, {have - size:+,}", flush=True)
    print(f"  The model file is only read; the copy goes in {folder} and is deleted at the end", flush=True)
    for p in leftovers(folder, source.name):
        try:
            p.unlink()
        except OSError:
            pass                                         # a copy an earlier run could not delete does not stop this one

    need = have
    free = free_bytes(folder)
    if free < need:
        raise SystemExit(f"{folder} has only {free:,} bytes free and this run needs {need:,} (one copy): "
                         "free some space, or point --work-dir at a roomier drive")

    copy = copied_to(source, folder, have)
    print(f"  Copied once: {copy.name} ({have:,} bytes)", flush=True)
    try:
        run_cases(copy, source, size, sha, cases, have)
    finally:
        try:
            copy.unlink()
            print("  Copy deleted; the model file was never written to", flush=True)
        except OSError as exc:
            print(f"  [!]  the copy could not be deleted ({exc}); delete {copy.name} by hand. "
                  "The model file was never written to", flush=True)
    return 0


def describe(length: int, have: int) -> str:
    """How far this case's file is from the model file's own length."""
    diff = length - have
    if diff == 0:
        return "the file as it is"
    return f"the file {abs(diff):,} bytes {'longer' if diff > 0 else 'shorter'}"


def drill_fake() -> int:
    """No model file: small GGUFs built here go through the same checks, to show what each one decides."""
    import setup

    size = 1_000_000_000                    # a published length that has nothing to do with these files
    sha = hashlib.sha256(b"drill").hexdigest()
    folder = Path(tempfile.mkdtemp(prefix="strata-drill-"))
    print(f"GGUFs built here (no network, no model file), published length {size:,}", flush=True)
    plan = (("whole, 96 extra bytes of padding at the end (another build)", tiny_gguf(2, 96), True),
            ("whole, exactly its own length", tiny_gguf(2, 0), True),
            ("cut short by 40 bytes", tiny_gguf(2, 0)[:-40], False),
            ("one tensor, 96 extra bytes (check_shards' single-tensor rule)", tiny_gguf(1, 96), False),
            ("not a GGUF file", b"not a gguf file at all", False))
    try:
        for index, (label, data, expect_ok) in enumerate(plan, 1):
            path = folder / "fake.gguf"
            path.write_bytes(data)
            print(f"\nCase {index}: {label}", flush=True)
            accepted, reason, said = check(path, size, sha, False, setup)
            if said:
                print(f"  setup said: {said}", flush=True)
            RESULTS.append((index, label, accepted, reason, expect_ok))
            print(verdict(index, accepted, expect_ok), flush=True)
            if accepted != expect_ok:
                print(f"      this case expects setup to {'accept' if expect_ok else 'refuse'}"
                      + (f"; setup said: {reason}" if reason else ""), flush=True)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        print(f"\n  Temporary folder {folder} deleted", flush=True)
    return 0


def summary() -> None:
    """The last block: what each case expected beside what setup did."""
    print("\n" + "=" * 60, flush=True)
    wrong = 0
    for index, what, accepted, reason, expected in RESULTS:
        ok = accepted == expected
        mark = "✅" if ok else "❌"
        print(f"  {mark} Case {index}: {what} (expected {'accepted' if expected else 'refused'}, "
              f"setup {'accepted' if accepted else 'refused'})", flush=True)
        if not ok:
            wrong += 1
            if reason:
                print(f"      setup said: {reason}", flush=True)
    print("=" * 60, flush=True)
    print(f"  {len(RESULTS)} cases, {len(RESULTS) - wrong} as expected"
          + (f", {wrong} NOT as expected" if wrong else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, help="the GGUF model file (read only, never deleted or changed)")
    ap.add_argument("--size", type=int, help="published length in bytes; without it, looked up in setup.py by file name")
    ap.add_argument("--sha256", help="published SHA-256; without it, looked up in setup.py by file name")
    ap.add_argument("--label", help="which table to use when the name is in several (gyro, unsloth)")
    ap.add_argument("--cases", default="-96,-2,all",
                    help="how many bytes to cut from the copy for each case; all = as it is. Default %(default)s")
    ap.add_argument("--work-dir", type=Path,
                    help="where the copy goes; default is the model file's folder (needs as much free space)")
    a = ap.parse_args()

    if a.file is None:
        code = drill_fake()
        summary()
        return code

    source = a.file
    if not source.is_file():
        raise SystemExit(f"no such file: {source}")
    found = None if (a.size and a.sha256) else pinned(a.label, source.name)
    size = a.size or (found[0] if found else None)
    sha = a.sha256 or (found[1] if found else None)
    if size is None:
        raise SystemExit(f"{source.name} is not in one of setup.py's tables: give --size and --sha256")
    if sha is None:
        print("  [!]  no published SHA-256: it is computed when the length matches (minutes on a big file)", flush=True)
        sha = "0" * 64

    have = source.stat().st_size
    cases = []
    for part in a.cases.split(","):
        part = part.strip()
        try:
            length = have if part == "all" else have + int(part)
        except ValueError:
            raise SystemExit(f"--cases: {part!r} is neither a whole number nor all") from None
        if length > have:
            print(f"  [!]  skipping {part}: the drill only reshapes bytes the file already has, it never "
                  "writes data into a copy", flush=True)
            continue
        if length <= 0:
            continue
        cases.append((part, length))

    folder = a.work_dir or source.parent
    if not folder.is_dir():
        raise SystemExit(f"no such folder: {folder}")
    code = drill_real(source, size, sha, cases, folder)
    summary()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
