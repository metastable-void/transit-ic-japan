#!/usr/bin/env python3
"""Dump FeliCa system 0x0003 service information using nfcpy."""

from __future__ import annotations

import argparse
import itertools
import struct
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import nfc
import nfc.tag
from nfc.tag import TagCommandError
from nfc.tag.tt3 import (
    BlockCode,
    DATA_SIZE_ERROR,
    ServiceCode,
    Type3Tag,
    Type3TagCommandError,
)


FELICA_SYSTEM_CODE = 0x0003
DEFAULT_POLL_INTERVAL = 0.2  # seconds between sense attempts
DEFAULT_POLL_TRIES = 0  # 0 => keep waiting forever
MAX_SERVICE_NUMBER = 1 << 10  # 10-bit service number space
REQUEST_SERVICE_CHUNK = 32
SERVICE_ATTRIBUTES_TO_SCAN = tuple(range(0x08, 0x18))


@dataclass
class ServiceInfo:
    index: Optional[int]
    service_code: ServiceCode
    raw_code: int
    key_version: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump all FeliCa (system code 0x0003) services readable without keys."
    )
    parser.add_argument(
        "--device",
        default="usb",
        dest="device",
        help="nfcpy ContactlessFrontend resource path (default: usb)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds to wait between polling attempts (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-tries",
        type=int,
        default=DEFAULT_POLL_TRIES,
        help="Number of polling iterations before giving up (0 waits indefinitely).",
    )
    return parser.parse_args()


def _build_sensf_req() -> bytearray:
    """Create a SENSF_REQ payload that selects system code 0x0003."""
    # Format: 0x00 + system code (big-endian) + request code + time slot.
    return bytearray(b"\x00" + struct.pack(">HBB", FELICA_SYSTEM_CODE, 0x00, 0x00))


def _make_remote_target(bit_rate: str) -> nfc.clf.RemoteTarget:
    target = nfc.clf.RemoteTarget(bit_rate)
    target.sensf_req = _build_sensf_req()
    return target


def wait_for_felica_target(
    clf: nfc.ContactlessFrontend, poll_tries: int, poll_interval: float
) -> Optional[nfc.clf.RemoteTarget]:
    tries_remaining = poll_tries if poll_tries > 0 else None
    while tries_remaining is None or tries_remaining > 0:
        targets = [_make_remote_target("212F"), _make_remote_target("424F")]
        target = clf.sense(*targets, iterations=1, interval=poll_interval)
        if target is not None:
            return target
        if tries_remaining is not None:
            tries_remaining -= 1
    return None


def select_system(tag: Type3Tag) -> Tuple[bytearray, bytearray, int]:
    """Re-select the card for the target system code and return identifiers."""
    poll_result = tag.polling(FELICA_SYSTEM_CODE, request_code=1)
    idm, pmm = poll_result[0], poll_result[1]
    system_code = FELICA_SYSTEM_CODE
    if len(poll_result) > 2 and poll_result[2]:
        system_code = struct.unpack(">H", poll_result[2])[0]
    tag.idm = idm
    tag.pmm = pmm
    tag.sys = system_code
    return idm, pmm, system_code


def describe_error(error: Type3TagCommandError) -> str:
    errno = getattr(error, "errno", None)
    if errno is None:
        return str(error)
    if errno > 0x00FF:
        return f"card returned status 0x{errno:04X}"
    return f"command failed (errno=0x{errno:02X})"


def discover_services(tag: Type3Tag) -> List[ServiceInfo]:
    """Return a list of ServiceInfo entries for the active system."""
    services: List[ServiceInfo] = []
    if hasattr(tag, "search_service_code"):
        try:
            services = _discover_services_with_search(tag)
        except Type3TagCommandError as exc:
            if getattr(exc, "errno", None) != 0xA0:
                raise
        except RuntimeError:
            services = []
    if not services:
        services = _discover_services_bruteforce(tag)
    if not services:
        raise RuntimeError("Service discovery commands not supported by this tag")
    if any(info.key_version is None for info in services):
        _populate_key_versions(tag, services)
    return services


def _discover_services_with_search(tag: Type3Tag) -> List[ServiceInfo]:
    services: List[ServiceInfo] = []
    seen_codes: set[int] = set()

    for service_index in itertools.count():
        if service_index >= 0x10000:
            break
        result = tag.search_service_code(service_index)
        if result is None:
            break
        if len(result) != 1:
            continue
        raw_code = result[0]
        if raw_code in seen_codes:
            continue
        seen_codes.add(raw_code)
        sc = ServiceCode(raw_code >> 6, raw_code & 0x3F)
        services.append(ServiceInfo(index=service_index, service_code=sc, raw_code=raw_code))
    return services


def _chunked(iterable, size: int):
    chunk: List = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _request_service_raw(tag: Type3Tag, service_list: List[ServiceCode]) -> List[int]:
    if not service_list:
        return []
    a = tag.pmm[2] & 7
    b = tag.pmm[2] >> 3 & 7
    e = tag.pmm[2] >> 6
    timeout = 302e-6 * ((b + 1) * len(service_list) + a + 1) * 4**e
    payload = bytearray([len(service_list)]) + b"".join(sc.pack() for sc in service_list)
    response = tag.send_cmd_recv_rsp(0x02, payload, timeout, check_status=False)
    expected_length = 1 + len(service_list) * 2
    if len(response) != expected_length:
        raise Type3TagCommandError(DATA_SIZE_ERROR)
    return [
        struct.unpack("<H", response[i : i + 2])[0]
        for i in range(1, expected_length, 2)
    ]


def request_service(tag: Type3Tag, service_list: List[ServiceCode]) -> List[int]:
    if hasattr(tag, "request_service"):
        return tag.request_service(service_list)
    return _request_service_raw(tag, service_list)


def _populate_key_versions(tag: Type3Tag, services: List[ServiceInfo]) -> None:
    raw_map = {info.raw_code: info for info in services}
    service_codes = [info.service_code for info in services if info.key_version is None]
    for chunk in _chunked(service_codes, REQUEST_SERVICE_CHUNK):
        try:
            key_versions = request_service(tag, chunk)
        except Type3TagCommandError as exc:
            if getattr(exc, "errno", None) and exc.errno > 0x00FF:
                continue
            raise
        for sc, version in zip(chunk, key_versions):
            if version == 0xFFFF:
                continue
            info = raw_map.get(int(sc))
            if info:
                info.key_version = version


def _discover_services_bruteforce(tag: Type3Tag) -> List[ServiceInfo]:
    services: List[ServiceInfo] = []
    seen_codes: set[int] = set()

    for attribute in SERVICE_ATTRIBUTES_TO_SCAN:
        for number_chunk in _chunked(range(MAX_SERVICE_NUMBER), REQUEST_SERVICE_CHUNK):
            service_list = [ServiceCode(number, attribute) for number in number_chunk]
            try:
                key_versions = request_service(tag, service_list)
            except Type3TagCommandError as exc:
                if getattr(exc, "errno", None) and exc.errno > 0x00FF:
                    continue
                raise
            for sc, version in zip(service_list, key_versions):
                if version == 0xFFFF:
                    continue
                raw_code = int(sc)
                if raw_code in seen_codes:
                    continue
                seen_codes.add(raw_code)
                services.append(
                    ServiceInfo(
                        index=None,
                        service_code=ServiceCode(sc.number, sc.attribute),
                        raw_code=raw_code,
                        key_version=version,
                    )
                )

    services.sort(key=lambda info: info.raw_code)
    return services


def describe_service_flags(service_code: ServiceCode) -> Dict[str, object]:
    attr = service_code.attribute
    flags: Dict[str, object] = {
        "attribute": attr,
        "requires_key": (attr & 0x01) == 0,
        "type": "Unknown",
        "readable": None,
        "writeable": None,
    }

    family = (attr >> 2) & 0x0F
    if family == 0b0010:
        flags["type"] = "Random"
        flags["readable"] = True
        flags["writeable"] = (attr & 0x02) == 0
    elif family == 0b0011:
        flags["type"] = "Cyclic"
        flags["readable"] = True
        flags["writeable"] = (attr & 0x02) == 0
    elif family == 0b0100:
        flags["type"] = "Purse"
        flags["readable"] = attr in (0x16, 0x17)
        flags["writeable"] = attr in (0x10, 0x11, 0x12, 0x13, 0x14, 0x15)
        flags["operation"] = {
            0x10: "Direct",
            0x11: "Direct",
            0x12: "Cashback",
            0x13: "Cashback",
            0x14: "Decrement",
            0x15: "Decrement",
            0x16: "Read Only",
            0x17: "Read Only",
        }.get(attr, "Unknown")
    return flags


def should_dump_blocks(flags: Dict[str, object]) -> bool:
    if flags["type"] not in {"Random", "Cyclic"}:
        return False
    if flags["requires_key"]:
        return False
    return bool(flags.get("readable"))


def read_service_blocks(tag: Type3Tag, service_code: ServiceCode) -> List[Tuple[int, bytes]]:
    blocks: List[Tuple[int, bytes]] = []
    for block_index in itertools.count():
        if block_index >= 0x10000:
            break
        block = BlockCode(block_index, service=0)
        try:
            data = tag.read_without_encryption([service_code], [block])
        except Type3TagCommandError as exc:
            if exc.errno and exc.errno > 0x00FF:
                break
            raise
        else:
            blocks.append((block_index, bytes(data)))
    return blocks


def formatted_block_line(block_index: int, data: bytes) -> str:
    hex_part = " ".join(f"{byte:02X}" for byte in data)
    ascii_part = "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in data)
    return f"      {block_index:04X}: {hex_part} |{ascii_part}|"


def get_key_version(tag: Type3Tag, service_code: ServiceCode) -> Optional[int]:
    try:
        response = request_service(tag, [service_code])
    except TagCommandError:
        return None
    return response[0] if response else None


def process_tag(tag: Type3Tag) -> int:
    try:
        idm, pmm, system_code = select_system(tag)
    except Type3TagCommandError as exc:
        print(
            f"Failed to activate system 0x{FELICA_SYSTEM_CODE:04X}: {describe_error(exc)}",
            file=sys.stderr,
        )
        return 1

    print(f"IDm        : {idm.hex().upper()}")
    print(f"PMm        : {pmm.hex().upper()}")
    print(f"System Code: 0x{system_code:04X}")

    try:
        services = discover_services(tag)
    except RuntimeError as exc:
        print(f"Service discovery not available: {exc}", file=sys.stderr)
        return 1
    except Type3TagCommandError as exc:
        print(f"Failed to enumerate services: {describe_error(exc)}", file=sys.stderr)
        return 1

    if not services:
        print("No services reported by the card.")
        return 0

    print(f"Found {len(services)} service(s):")
    for info in services:
        index_label = f"0x{info.index:04X}" if info.index is not None else "n/a"
        print(f"- Service index {index_label} / code 0x{info.raw_code:04X}")
        print(f"    {info.service_code}")

        flags = describe_service_flags(info.service_code)
        print(
            "    Attribute: 0x{attr:02X} (type={type}, key required={key})".format(
                attr=flags["attribute"],
                type=flags["type"],
                key="yes" if flags["requires_key"] else "no",
            )
        )

        if flags.get("readable") is not None or flags.get("writeable") is not None:
            print(
                "    Access   : read={read} write={write}".format(
                    read="yes" if flags.get("readable") else "no",
                    write="yes" if flags.get("writeable") else "no",
                )
            )

        if "operation" in flags:
            print(f"    Operation: {flags['operation']}")

        key_version = info.key_version
        if key_version is None:
            key_version = get_key_version(tag, info.service_code)
            info.key_version = key_version
        if key_version is not None:
            print(f"    Key Vers.: 0x{key_version:04X}")

        if should_dump_blocks(flags):
            try:
                blocks = read_service_blocks(tag, info.service_code)
            except Type3TagCommandError as exc:
                print(f"    Block dump failed: {describe_error(exc)}")
            else:
                if not blocks:
                    print("    Block dump: no readable blocks detected.")
                else:
                    print("    Blocks:")
                    for block_index, data in blocks:
                        print(formatted_block_line(block_index, data))
        else:
            print("    Block dump: skipped (requires keys or unsupported type).")

    return 0


def main() -> int:
    args = parse_args()
    try:
        clf = nfc.ContactlessFrontend(args.device)
    except IOError as exc:
        print(f"Unable to open NFC device '{args.device}': {exc}", file=sys.stderr)
        return 1

    print(f"Waiting for FeliCa card (system code 0x{FELICA_SYSTEM_CODE:04X})...")
    with clf:
        target = wait_for_felica_target(clf, args.poll_tries, args.poll_interval)
        if target is None:
            print("No FeliCa card detected.")
            return 1

        try:
            tag = nfc.tag.activate(clf, target)
        except Exception as exc:  # noqa: BLE001 - nfcpy may raise various errors
            print(f"Failed to activate tag: {exc}", file=sys.stderr)
            return 1

        if not isinstance(tag, Type3Tag):
            print("Detected tag is not a FeliCa / Type 3 tag.", file=sys.stderr)
            return 1

        try:
            return process_tag(tag)
        finally:
            closer = getattr(tag, "close", None)
            if callable(closer):
                closer()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
