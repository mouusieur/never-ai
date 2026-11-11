"""Deterministic Pure-Python Native Windows x64 Self-Extracting GUI Executable Builder (`NeverAI-Setup.exe`).

Builds a 100% genuine 64-bit Windows PE32+ (`IMAGE_FILE_MACHINE_AMD64`, `WINDOWS_GUI` subsystem,
zero console flash) stub (`2,560` bytes) followed at exact byte offset `2560` by the complete
`NeverAI-Installer-Windows-x64.zip` archive.

When double-clicked on Windows 10/11:
1. Calls `KERNEL32!GetModuleFileNameA(NULL, pathBuf, 500)` to locate its own `.exe` path (even if renamed).
2. Calls `KERNEL32!SetEnvironmentVariableA("NEVER_AI_SFX_EXE", pathBuf)`.
3. Calls `KERNEL32!WinExec(psCmd, SW_HIDE)` which slices the embedded ZIP at offset `2560` in memory,
   extracts to `%TEMP%\\NeverAI-Setup`, and launches the Monolith Noir Graphical Setup Wizard (`setup_wizard.ps1`).
4. Calls `KERNEL32!ExitProcess(0)`.
"""

from __future__ import annotations

from pathlib import Path
import struct

from never_ai.desktop.installer_builder import build_windows_installer_zip


SFX_ZIP_OFFSET = 0x0A00  # 2560 bytes exact PE32+ stub size


def _pack_rip_rel_lea(reg_opcode_byte2: int, instr_rva: int, target_rva: int) -> bytes:
    """Encode `48 8D <modrm> <disp32>` (7 bytes) where RIP-relative base is `instr_rva + 7`."""
    disp = target_rva - (instr_rva + 7)
    return b"\x48\x8d" + bytes([reg_opcode_byte2]) + struct.pack("<i", disp)


def _pack_rip_rel_call_iat(instr_rva: int, iat_entry_rva: int) -> bytes:
    """Encode `FF 15 <disp32>` (6 bytes) `call qword ptr [rip + disp32]` where base is `instr_rva + 6`."""
    disp = iat_entry_rva - (instr_rva + 6)
    return b"\xff\x15" + struct.pack("<i", disp)


def build_pe64_sfx_stub() -> bytes:
    """Construct the exact 2,560-byte (`0x0A00`) PE32+ Windows x64 GUI self-extractor stub."""
    # -------------------------------------------------------------------------
    # 1. Build .rdata / .idata section (VirtualAddress = 0x2000, RawSize = 0x0600)
    # -------------------------------------------------------------------------
    rdata = bytearray(0x0600)

    def put_rdata(rva: int, data: bytes) -> None:
        off = rva - 0x2000
        rdata[off : off + len(data)] = data

    # Import Directory Table at 0x2000 (2 descriptors * 20 bytes = 40 bytes)
    # Descriptor 0: ILT=0x2028, TimeDate=0, Fwd=0, Name=0x2080 ("KERNEL32.dll"), IAT=0x2058
    put_rdata(0x2000, struct.pack("<IIIII", 0x2028, 0, 0, 0x2080, 0x2058))
    # Descriptor 1 at 0x2014 is all zeros (null terminator)

    # Hint/Name entries
    put_rdata(0x2080, b"KERNEL32.dll\x00")
    put_rdata(0x2090, b"\x00\x00GetModuleFileNameA\x00")
    put_rdata(0x20A8, b"\x00\x00SetEnvironmentVariableA\x00")
    put_rdata(0x20C4, b"\x00\x00WinExec\x00")
    put_rdata(0x20D0, b"\x00\x00ExitProcess\x00")

    # ILT (0x2028) & IAT (0x2058): 4 64-bit RVAs + 64-bit zero terminator
    thunks = struct.pack("<QQQQQ", 0x2090, 0x20A8, 0x20C4, 0x20D0, 0)
    put_rdata(0x2028, thunks)
    put_rdata(0x2058, thunks)

    # Environment variable name & PowerShell in-memory SFX command
    put_rdata(0x20E0, b"NEVER_AI_SFX_EXE\x00")
    ps_cmd = (
        "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -STA -Command "
        "\"$b=[IO.File]::ReadAllBytes($env:NEVER_AI_SFX_EXE);"
        f"$ms=New-Object IO.MemoryStream($b,{SFX_ZIP_OFFSET},($b.Length-{SFX_ZIP_OFFSET}));"
        "Add-Type -A System.IO.Compression.FileSystem;"
        "$d=Join-Path $env:TEMP 'NeverAI-Setup';"
        "Remove-Item $d -Recurse -Force -EA 0;"
        "$z=New-Object IO.Compression.ZipArchive($ms);"
        "[IO.Compression.ZipFileExtensions]::ExtractToDirectory($z,$d);"
        "& (Join-Path $d 'setup_wizard.ps1')\"\x00"
    ).encode("ascii")
    assert 0x2100 + len(ps_cmd) < 0x2400, "PowerShell SFX command exceeds reserved slot"
    put_rdata(0x2100, ps_cmd)
    # 0x2400 .. 0x25FF (512 bytes) stays zeroed as writable pathBuf for GetModuleFileNameA

    # -------------------------------------------------------------------------
    # 2. Build .text x86-64 machine code (VirtualAddress = 0x1000, RawSize = 0x0200)
    # -------------------------------------------------------------------------
    code = bytearray()
    cur_rva = 0x1000

    def emit(b: bytes) -> None:
        nonlocal cur_rva
        code.extend(b)
        cur_rva += len(b)

    # sub rsp, 0x28  (Win64 ABI: 16-byte stack alignment + 32-byte shadow space)
    emit(b"\x48\x83\xec\x28")
    # xor ecx, ecx   (hModule = NULL)
    emit(b"\x31\xc9")
    # lea rdx, [rip + pathBuf (0x2400)]
    emit(_pack_rip_rel_lea(0x15, cur_rva, 0x2400))
    # mov r8d, 500
    emit(b"\x41\xb8\xf4\x01\x00\x00")
    # call [rip + IAT_GetModuleFileNameA (0x2058)]
    emit(_pack_rip_rel_call_iat(cur_rva, 0x2058))

    # lea rcx, [rip + "NEVER_AI_SFX_EXE" (0x20E0)]
    emit(_pack_rip_rel_lea(0x0D, cur_rva, 0x20E0))
    # lea rdx, [rip + pathBuf (0x2400)]
    emit(_pack_rip_rel_lea(0x15, cur_rva, 0x2400))
    # call [rip + IAT_SetEnvironmentVariableA (0x2060)]
    emit(_pack_rip_rel_call_iat(cur_rva, 0x2060))

    # lea rcx, [rip + ps_cmd (0x2100)]
    emit(_pack_rip_rel_lea(0x0D, cur_rva, 0x2100))
    # xor edx, edx   (uCmdShow = SW_HIDE = 0)
    emit(b"\x31\xd2")
    # call [rip + IAT_WinExec (0x2068)]
    emit(_pack_rip_rel_call_iat(cur_rva, 0x2068))

    # xor ecx, ecx   (uExitCode = 0)
    emit(b"\x31\xc9")
    # call [rip + IAT_ExitProcess (0x2070)]
    emit(_pack_rip_rel_call_iat(cur_rva, 0x2070))
    # int3 padding
    emit(b"\xcc")

    text_sec = bytes(code).ljust(0x0200, b"\x00")

    # -------------------------------------------------------------------------
    # 3. Build DOS + PE32+ Headers (RawSize = 0x0200)
    # -------------------------------------------------------------------------
    headers = bytearray(0x0200)
    # Minimal DOS Header ('MZ' at 0x00, e_lfanew = 0x80 at 0x3C)
    headers[0:2] = b"MZ"
    struct.pack_into("<I", headers, 0x3C, 0x80)
    dos_msg = b"This program requires Microsoft Windows x64.\r\n$"
    headers[0x40 : 0x40 + len(dos_msg)] = dos_msg

    # PE Signature at 0x80
    headers[0x80:0x84] = b"PE\x00\x00"

    # COFF File Header (20 bytes at 0x84)
    struct.pack_into(
        "<HHIIIHH",
        headers,
        0x84,
        0x8664,  # Machine: AMD64
        2,       # NumberOfSections: .text, .rdata
        0,       # TimeDateStamp
        0,       # PointerToSymbolTable
        0,       # NumberOfSymbols
        240,     # SizeOfOptionalHeader (PE32+)
        0x0023,  # Characteristics: RELOCS_STRIPPED | EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE
    )

    # Optional Header PE32+ (240 bytes at 0x98 .. 0x188)
    opt = bytearray(240)
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        opt,
        0,
        0x020B,         # Magic: PE32+ (64-bit)
        14, 0,          # Linker version
        0x0200,         # SizeOfCode
        0x0600,         # SizeOfInitializedData
        0,              # SizeOfUninitializedData
        0x1000,         # AddressOfEntryPoint (.text)
        0x1000,         # BaseOfCode
        0x140000000,    # ImageBase
        0x1000,         # SectionAlignment
        0x0200,         # FileAlignment
        6, 0,           # OS Version (Windows Vista/7/10/11+)
        1, 4,           # Image Version 1.4
        6, 0,           # Subsystem Version 6.0
        0,              # Win32VersionValue
        0x3000,         # SizeOfImage (headers + .text + .rdata = 3 pages)
        0x0200,         # SizeOfHeaders
        0,              # CheckSum
        2,              # Subsystem: IMAGE_SUBSYSTEM_WINDOWS_GUI (2)
        0x8100,         # DllCharacteristics: NX_COMPAT | TERMINAL_SERVER_AWARE (RIP-relative, fixed base)
        0x100000,       # SizeOfStackReserve
        0x1000,         # SizeOfStackCommit
        0x100000,       # SizeOfHeapReserve
        0x1000,         # SizeOfHeapCommit
        0,              # LoaderFlags
        16,             # NumberOfRvaAndSizes
    )
    # Data Directories start at offset 112 inside OptionalHeader (16 * 8 = 128 bytes)
    # Directory[1] = Import Table (RVA 0x2000, Size 40)
    struct.pack_into("<II", opt, 112 + 1 * 8, 0x2000, 40)
    # Directory[12] = IAT (RVA 0x2058, Size 40)
    struct.pack_into("<II", opt, 112 + 12 * 8, 0x2058, 40)
    headers[0x98 : 0x98 + 240] = opt

    # Section Headers at 0x188 (2 * 40 bytes)
    # Section 0: .text
    struct.pack_into(
        "<8sIIIIIIHHI",
        headers,
        0x188,
        b".text\x00\x00\x00",
        len(code),   # VirtualSize
        0x1000,      # VirtualAddress
        0x0200,      # SizeOfRawData
        0x0200,      # PointerToRawData
        0, 0, 0, 0,
        0x60000020,  # CNT_CODE | MEM_EXECUTE | MEM_READ
    )
    # Section 1: .rdata (readable + writable for IAT & pathBuf)
    struct.pack_into(
        "<8sIIIIIIHHI",
        headers,
        0x188 + 40,
        b".rdata\x00\x00",
        0x0600,      # VirtualSize
        0x2000,      # VirtualAddress
        0x0600,      # SizeOfRawData
        0x0400,      # PointerToRawData
        0, 0, 0, 0,
        0xC0000040,  # CNT_INITIALIZED_DATA | MEM_READ | MEM_WRITE
    )

    stub = bytes(headers) + text_sec + bytes(rdata)
    assert len(stub) == SFX_ZIP_OFFSET, f"Expected {SFX_ZIP_OFFSET} bytes, got {len(stub)}"
    return stub


def build_windows_setup_exe(repo_root: str | Path) -> bytes:
    """Build the complete single-file `NeverAI-Setup.exe` (PE32+ x64 GUI stub + embedded ZIP)."""
    stub = build_pe64_sfx_stub()
    zip_payload = build_windows_installer_zip(repo_root)
    return stub + zip_payload
