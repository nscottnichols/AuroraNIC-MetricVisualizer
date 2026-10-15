"""Bounded binary reader for the explicitly validated LTTng-UST CTF 1.8 layout.

This is not a general CTF reader. Metadata is parsed and checked before any data
is decoded. Unsupported layouts must use Babeltrace 2; corruption/loss is never
silently skipped. Supports the byte-aligned, little-endian large event headers
and flat string/integer/sequence sampling payloads emitted by this THAPI capture.
"""
from __future__ import annotations
import mmap
from pathlib import Path
import re
import struct
from typing import Callable
import uuid

VERSION = "thapi-lttng-ctf/1"
CXI_EVENT = "lttng_ust_cxi_sampling:cxi"
MAX_METADATA = 16 * 1024 * 1024
MAX_EVENTS = 10_000_000


class PortableCtfError(ValueError):
    pass


def _number(text: str, key: str, default=None):
    m = re.search(r"\b" + re.escape(key) + r"\s*=\s*(-?\d+)\s*;", text)
    if m:
        return int(m.group(1))
    if default is not None:
        return default
    raise PortableCtfError(f"Missing CTF metadata field {key}; use Babeltrace 2.")


def _string(text: str, key: str):
    m = re.search(r"\b" + re.escape(key) + r'\s*=\s*"([^"\\]*)"\s*;', text)
    if not m:
        raise PortableCtfError(f"Missing/unsupported CTF string {key}.")
    return m.group(1)


def _flat_fields(text):
    pattern = re.compile(r"\s*(?:integer\s*\{([^{}]*)\}|(string))\s+(\w+)\s*(?:\[\s*(\w+)\s*\])?\s*;")
    fields = []
    pos = 0
    while text[pos:].strip():
        match = pattern.match(text, pos)
        if not match:
            raise PortableCtfError("Unsupported sampling payload layout; install Babeltrace 2.")
        body, string, name, length = match.groups()
        if string:
            if length:
                raise PortableCtfError("String arrays require Babeltrace 2.")
            fields.append((name, 'string', 0, None))
        else:
            size = _number(body, 'size'); align = _number(body, 'align', 1)
            if size not in {8,16,32,64} or align != 8:
                raise PortableCtfError("Non-byte-aligned payload requires Babeltrace 2.")
            signed = bool(re.search(r'\bsigned\s*=\s*true', body))
            fields.append((name, 'signed' if signed else 'unsigned', size//8, length))
        pos = match.end()
    return fields


def _metadata(path: Path):
    if path.stat().st_size > MAX_METADATA:
        raise PortableCtfError("CTF metadata exceeds the 16 MiB reader limit.")
    text = path.read_text(encoding='utf-8')
    trace = re.search(r'(?ms)^trace\s*\{(.*?)^\};', text)
    if not trace or 'CTF 1.8' not in text[:80] or not re.search(r'byte_order\s*=\s*le', trace[1]):
        raise PortableCtfError("Portable reader requires little-endian CTF 1.8; use Babeltrace 2.")
    if _number(trace[1], 'major') != 1 or _number(trace[1], 'minor') != 8:
        raise PortableCtfError("Unsupported CTF version.")
    # Validate the exact packet header and stream structures, not just the event ID.
    compact = lambda s: re.sub(r'\s+', '', s)
    header = re.search(r'packet.header\s*:=\s*struct\s*\{(.*?)\};', trace[1], re.S)
    expected_header = '''integer {size=32;align=8;base=16;} magic;
    integer {size=8;align=8;base=16;} uuid[16]; integer {size=32;align=8;} stream_id;
    integer {size=64;align=8;} stream_instance_id;'''
    if not header or compact(header[1]) != compact(expected_header):
        raise PortableCtfError("Unsupported packet header; use Babeltrace 2.")
    clocks = re.findall(r'(?ms)^clock\s*\{(.*?)^\};', text)
    if len(clocks)!=1 or _number(clocks[0],'freq')!=1_000_000_000:
        raise PortableCtfError("Portable reader requires one 1 GHz clock; use Babeltrace 2.")
    clock = _string(clocks[0], 'name')
    offset = _number(clocks[0], 'offset', 0) + _number(clocks[0], 'offset_s', 0)*1_000_000_000
    schemas = {}
    for block in re.findall(r'(?ms)^event\s*\{(.*?)^\};', text):
        sid = _number(block,'stream_id')
        # Compile only streams containing CXI, after finding them all below.
        schemas.setdefault(sid, {})[_number(block,'id')] = (_string(block,'name'),block)
    target_ids = [sid for sid, events in schemas.items() if any(x[0]==CXI_EVENT for x in events.values())]
    if not target_ids:
        raise PortableCtfError(f"No {CXI_EVENT} event schema in {path}.")
    streams = {_number(block,'id'):block for block in re.findall(r'(?ms)^stream\s*\{(.*?)^\};',text)}
    compiled = {}
    expected_stream = '''id=SID;
    event.header := struct {
      enum : integer {size=16;align=8;} {"extended"=65535,"compact"=0...65534,} id;
      variant <id> {struct {integer {size=32;align=8;map=clock.CLOCK.value;} timestamp;} compact;
      struct {integer {size=32;align=8;} id;integer {size=64;align=8;map=clock.CLOCK.value;} timestamp;} extended;} v;};
    packet.context := struct {
      integer {size=64;align=8;map=clock.CLOCK.value;} timestamp_begin;
      integer {size=64;align=8;map=clock.CLOCK.value;} timestamp_end;
      integer {size=64;align=8;} content_size;integer {size=64;align=8;} packet_size;
      integer {size=64;align=8;} packet_seq_num;integer {size=64;align=8;} events_discarded;
      integer {size=32;align=8;} cpu_id;};
    event.context := struct {integer {size=32;align=8;signed=true;} _vpid;
      integer {size=32;align=8;signed=true;} _vtid;};'''
    for sid in target_ids:
        expected = expected_stream.replace('SID',str(sid)).replace('CLOCK',clock)
        if compact(streams.get(sid,'')) != compact(expected):
            raise PortableCtfError("Unsupported stream header/context; use Babeltrace 2.")
        for eid,(name,block) in schemas[sid].items():
            payload = re.search(r'fields\s*:=\s*struct\s*\{(.*)\};\s*$',block,re.S)
            if not payload:
                raise PortableCtfError("Missing sampling payload schema.")
            compiled[(sid,eid)] = (name,_flat_fields(payload[1]))
    return _string(text,'hostname'),uuid.UUID(_string(trace[1],'uuid')),offset,set(target_ids),compiled


def extract_portable_ctf(path, *, should_cancel: Callable[[], bool] | None = None):
    """Return structured CXI events and a count of other decoded sampling events."""
    root = Path(path).expanduser().resolve(strict=True)
    metadata = [root] if root.is_file() and root.name=='metadata' else sorted(root.rglob('metadata'))
    if not metadata:
        raise PortableCtfError("Choose a CTF directory containing metadata and stream files.")
    events=[]; filtered=0
    for meta in metadata:
        if should_cancel and should_cancel():
            raise PortableCtfError('CTF import cancelled.')
        host, trace_uuid, offset, targets, schemas = _metadata(meta)
        for file in sorted(meta.parent.iterdir()):
            if not file.is_file() or file.name=='metadata' or file.stat().st_size < 32:
                continue
            with file.open('rb') as fp, mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ) as data:
                if data[:4] != b'\xc1\x1f\xfc\xc1':
                    continue
                if struct.unpack_from('<I',data,20)[0] not in targets:
                    continue
                pos=0; last_seq=None
                while pos<len(data):
                    if should_cancel and should_cancel():
                        raise PortableCtfError('CTF import cancelled.')
                    if len(data)-pos<84:
                        raise PortableCtfError(f'Truncated CTF packet in {file.name}.')
                    magic,uid,sid,instance=struct.unpack_from('<I16sIQ',data,pos)
                    if magic!=0xc1fc1fc1 or uid!=trace_uuid.bytes or sid not in targets:
                        raise PortableCtfError(f'Packet identity mismatch in {file.name}.')
                    begin,end,content_bits,packet_bits,seq,discarded,cpu=struct.unpack_from('<QQQQQQI',data,pos+32)
                    if (content_bits%8 or packet_bits%8 or content_bits<84*8 or
                        content_bits>packet_bits or packet_bits==0 or pos+packet_bits//8>len(data) or end<begin):
                        raise PortableCtfError(f'Invalid/truncated packet sizes in {file.name}.')
                    if discarded or (last_seq is not None and seq!=last_seq+1):
                        raise PortableCtfError('Lost/discarded sampling records detected; use a loss-aware Babeltrace import.')
                    last_seq=seq; limit=pos+content_bits//8; cursor=pos+84; previous=begin
                    while cursor<limit:
                        if cursor+6>limit:
                            raise PortableCtfError('Truncated CTF event header.')
                        eid=struct.unpack_from('<H',data,cursor)[0];cursor+=2
                        if eid==65535:
                            if cursor+12>limit: raise PortableCtfError('Truncated extended event.')
                            eid,timestamp=struct.unpack_from('<IQ',data,cursor);cursor+=12
                        else:
                            low=struct.unpack_from('<I',data,cursor)[0];cursor+=4
                            timestamp=(previous & ~0xffffffff)|low
                            if timestamp<previous:timestamp+=1<<32
                        if not previous<=timestamp<=end or cursor+8>limit:
                            raise PortableCtfError('Invalid CTF event timestamp/context.')
                        previous=timestamp;pid,tid=struct.unpack_from('<ii',data,cursor);cursor+=8
                        if (sid,eid) not in schemas:raise PortableCtfError(f'Undeclared event ID {eid}.')
                        name,fields=schemas[(sid,eid)]; values={}
                        for field,kind,width,length in fields:
                            if kind=='string':
                                end_string=data.find(b'\0',cursor,limit)
                                if end_string<0:raise PortableCtfError('Unterminated CTF string.')
                                values[field]=data[cursor:end_string].decode('utf-8');cursor=end_string+1
                            else:
                                count=(int(length) if length.isdigit() else values.get(length)) if length else 1
                                if not isinstance(count,int) or count<0 or cursor+count*width>limit:
                                    raise PortableCtfError('Invalid/truncated CTF sequence.')
                                if not length:
                                    values[field]=int.from_bytes(data[cursor:cursor+width],'little',signed=kind=='signed')
                                cursor+=count*width
                        if name==CXI_EVENT:
                            if not {'_interface_name','_counter','_value'}<=values.keys():
                                raise PortableCtfError('Unsupported CXI payload.')
                            events.append({'name':'cxi','provider':'lttng_ust_cxi_sampling','kind':'raw_sampling',
                              'timestamp_ns':timestamp+offset,'hostname':host,
                              'collector':{'name':'lttng-ust-sampling','rank':None,
                                'process':f'trace={trace_uuid};pid={pid};tid={tid}'},
                              'payload':{'interface_name':values['_interface_name'],'counter':values['_counter'],'value':values['_value']}})
                            if len(events)>MAX_EVENTS:raise PortableCtfError('Trace exceeds the ten-million-event import limit.')
                        else:filtered+=1
                    if cursor!=limit:raise PortableCtfError('CTF content boundary mismatch.')
                    pos+=packet_bits//8
    if not events:
        raise PortableCtfError('No CXI counter samples in the chosen CTF directory.')
    return events,filtered
