#!/usr/bin/env python3
"""Read existing GGUF, add tokenizer metadata, write new GGUF."""
import struct, json, sys, os, shutil

GGUF_MAGIC = 0x46554747

GGUF_VALUE_UINT8   = 0
GGUF_VALUE_INT8    = 1
GGUF_VALUE_UINT16  = 2
GGUF_VALUE_INT16   = 3
GGUF_VALUE_UINT32  = 4
GGUF_VALUE_INT32   = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL    = 7
GGUF_VALUE_STRING  = 8
GGUF_VALUE_ARRAY   = 9
GGUF_VALUE_UINT64  = 10
GGUF_VALUE_INT64   = 11
GGUF_VALUE_FLOAT64 = 12

def read_generic(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))[0]

def write_generic(f, fmt, val):
    f.write(struct.pack(fmt, val))

def write_kv_str(f, key, val):
    bkey = key.encode('utf-8')
    bval = val.encode('utf-8')
    write_generic(f, 'Q', len(bkey))
    f.write(bkey)
    write_generic(f, 'I', GGUF_VALUE_STRING)
    write_generic(f, 'Q', len(bval))
    f.write(bval)

def write_kv_u32(f, key, val):
    bkey = key.encode('utf-8')
    write_generic(f, 'Q', len(bkey))
    f.write(bkey)
    write_generic(f, 'I', GGUF_VALUE_UINT32)
    write_generic(f, 'I', val)

def write_kv_f32(f, key, val):
    bkey = key.encode('utf-8')
    write_generic(f, 'Q', len(bkey))
    f.write(bkey)
    write_generic(f, 'I', GGUF_VALUE_FLOAT32)
    write_generic(f, 'f', val)

def write_kv_array(f, key, elem_type, items):
    bkey = key.encode('utf-8')
    write_generic(f, 'Q', len(bkey))
    f.write(bkey)
    write_generic(f, 'I', GGUF_VALUE_ARRAY)
    write_generic(f, 'I', elem_type)
    write_generic(f, 'Q', len(items))
    for item in items:
        if elem_type == GGUF_VALUE_INT32:
            write_generic(f, 'i', item)
        elif elem_type == GGUF_VALUE_FLOAT32:
            write_generic(f, 'f', item)
        elif elem_type == GGUF_VALUE_STRING:
            write_generic(f, 'Q', len(item))
            f.write(item.encode('utf-8') if isinstance(item, str) else item)
        else:
            raise ValueError(f"unsupported array elem type {elem_type}")

def main():
    src_path = sys.argv[1]
    dst_path = sys.argv[2] if len(sys.argv) > 2 else src_path + '.patched'
    tok_path = os.path.join(os.path.dirname(src_path), 'tokenizer.json')

    with open(tok_path) as f:
        tok_data = json.load(f)

    vocab = tok_data.get('model', {}).get('vocab', {})
    added_tokens = tok_data.get('added_tokens', [])
    
    # Sort vocab by id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    # Some tokens may be in added_tokens with same ID - merge
    token_texts = {}
    for token, tid in sorted_vocab:
        token_texts[tid] = token
    
    # Handle added tokens (they may have higher IDs)
    max_id = max(token_texts.keys()) if token_texts else 0
    for at in added_tokens:
        tid = at['id']
        token_texts[tid] = at['content']

    # Create sorted arrays
    ids = sorted(token_texts.keys())
    n_vocab = len(ids)

    with open(src_path, 'rb') as f_in, open(dst_path, 'wb') as f_out:
        # Read existing header
        magic = struct.unpack('I', f_in.read(4))[0]
        if magic != GGUF_MAGIC:
            print("Not a GGUF file", file=sys.stderr)
            sys.exit(1)
        
        version = struct.unpack('I', f_in.read(4))[0]
        n_tensors = struct.unpack('Q', f_in.read(8))[0]
        n_kv = struct.unpack('Q', f_in.read(8))[0]
        
        print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} KV entries")
        
        # Read all existing KV entries
        kv_entries = []
        for i in range(n_kv):
            key_len = struct.unpack('Q', f_in.read(8))[0]
            key = f_in.read(key_len).decode('utf-8')
            val_type = struct.unpack('I', f_in.read(4))[0]
            val_pos = f_in.tell()
            
            if val_type == GGUF_VALUE_UINT8:
                val = struct.unpack('B', f_in.read(1))[0]
            elif val_type == GGUF_VALUE_INT8:
                val = struct.unpack('b', f_in.read(1))[0]
            elif val_type == GGUF_VALUE_UINT16:
                val = struct.unpack('H', f_in.read(2))[0]
            elif val_type == GGUF_VALUE_INT16:
                val = struct.unpack('h', f_in.read(2))[0]
            elif val_type == GGUF_VALUE_UINT32:
                val = struct.unpack('I', f_in.read(4))[0]
            elif val_type == GGUF_VALUE_INT32:
                val = struct.unpack('i', f_in.read(4))[0]
            elif val_type == GGUF_VALUE_FLOAT32:
                val = struct.unpack('f', f_in.read(4))[0]
            elif val_type == GGUF_VALUE_BOOL:
                val = bool(struct.unpack('B', f_in.read(1))[0])
            elif val_type == GGUF_VALUE_UINT64:
                val = struct.unpack('Q', f_in.read(8))[0]
            elif val_type == GGUF_VALUE_INT64:
                val = struct.unpack('q', f_in.read(8))[0]
            elif val_type == GGUF_VALUE_FLOAT64:
                val = struct.unpack('d', f_in.read(8))[0]
            elif val_type == GGUF_VALUE_STRING:
                slen = struct.unpack('Q', f_in.read(8))[0]
                val = f_in.read(slen).decode('utf-8', errors='replace')
            elif val_type == GGUF_VALUE_ARRAY:
                elem_type = struct.unpack('I', f_in.read(4))[0]
                n_items = struct.unpack('Q', f_in.read(8))[0]
                arr = []
                for _ in range(n_items):
                    if elem_type == GGUF_VALUE_UINT32:
                        arr.append(struct.unpack('I', f_in.read(4))[0])
                    elif elem_type == GGUF_VALUE_INT32:
                        arr.append(struct.unpack('i', f_in.read(4))[0])
                    elif elem_type == GGUF_VALUE_FLOAT32:
                        arr.append(struct.unpack('f', f_in.read(4))[0])
                    elif elem_type == GGUF_VALUE_STRING:
                        sl = struct.unpack('Q', f_in.read(8))[0]
                        arr.append(f_in.read(sl).decode('utf-8'))
                    else:
                        print(f"Unknown array elem type {elem_type}", file=sys.stderr)
                        sys.exit(1)
                val = arr
            else:
                print(f"Unknown value type {val_type} for key {key}", file=sys.stderr)
                sys.exit(1)
            kv_entries.append((key, val_type, val))
        
        # Read tensor info
        tensor_infos = []
        alignment = 32
        for i in range(n_tensors):
            name_len = struct.unpack('Q', f_in.read(8))[0]
            name = f_in.read(name_len).decode('utf-8')
            ndim = struct.unpack('I', f_in.read(4))[0]
            dims = [struct.unpack('Q', f_in.read(8))[0] for _ in range(ndim)]
            ggml_type = struct.unpack('I', f_in.read(4))[0]
            offset = struct.unpack('Q', f_in.read(8))[0]
            tensor_infos.append((name, ndim, dims, ggml_type, offset))
            for kv in kv_entries:
                if kv[0] == 'general.alignment':
                    alignment = kv[2]
                    break
        
        hdr_end = f_in.tell()
        
        # Now rebuild the GGUF file with tokenizer metadata added
        # Count additional KV entries needed:
        # - tokenizer.ggml.tokens (array of int32)
        # - tokenizer.ggml.scores (array of float32) 
        # - tokenizer.ggml.token_<id>.text (string per token - too many)
        # Better: just write the tokens and scores arrays, plus bos/eos ids
        
        n_new_kv = len(kv_entries) + 4  # existing + tokens + scores + bos + eos
        
        # Write new header
        write_generic(f_out, 'I', GGUF_MAGIC)
        write_generic(f_out, 'I', version)
        write_generic(f_out, 'Q', n_tensors)
        write_generic(f_out, 'Q', n_new_kv)
        
        # Write existing KV entries
        for key, val_type, val in kv_entries:
            bkey = key.encode('utf-8')
            write_generic(f_out, 'Q', len(bkey))
            f_out.write(bkey)
            write_generic(f_out, 'I', val_type)
            if val_type in (GGUF_VALUE_UINT8,):
                write_generic(f_out, 'B', val)
            elif val_type == GGUF_VALUE_INT8:
                write_generic(f_out, 'b', val)
            elif val_type == GGUF_VALUE_UINT16:
                write_generic(f_out, 'H', val)
            elif val_type == GGUF_VALUE_INT16:
                write_generic(f_out, 'h', val)
            elif val_type == GGUF_VALUE_UINT32:
                write_generic(f_out, 'I', val)
            elif val_type == GGUF_VALUE_INT32:
                write_generic(f_out, 'i', val)
            elif val_type == GGUF_VALUE_FLOAT32:
                write_generic(f_out, 'f', val)
            elif val_type == GGUF_VALUE_BOOL:
                write_generic(f_out, 'B', int(val))
            elif val_type == GGUF_VALUE_UINT64:
                write_generic(f_out, 'Q', val)
            elif val_type == GGUF_VALUE_INT64:
                write_generic(f_out, 'q', val)
            elif val_type == GGUF_VALUE_FLOAT64:
                write_generic(f_out, 'd', val)
            elif val_type == GGUF_VALUE_STRING:
                bval = val.encode('utf-8')
                write_generic(f_out, 'Q', len(bval))
                f_out.write(bval)
            elif val_type == GGUF_VALUE_ARRAY:
                # Need to know elem_type - infer from first element type
                print(f"Array {key}: {len(val)} items", file=sys.stderr)
                # We'll skip arrays for now (not needed for tokenizer)
        
        # Write tokenizer metadata
        write_kv_u32(f_out, "tokenizer.ggml.bos_token_id", 120000)
        write_kv_u32(f_out, "tokenizer.ggml.eos_token_id", 120025)
        write_kv_str(f_out, "tokenizer.ggml.model", "gpt2")
        
        # Write token ID array
        bkey = b"tokenizer.ggml.tokens"
        write_generic(f_out, 'Q', len(bkey))
        f_out.write(bkey)
        write_generic(f_out, 'I', GGUF_VALUE_ARRAY)
        write_generic(f_out, 'I', GGUF_VALUE_INT32)
        write_generic(f_out, 'Q', n_vocab)
        for tid in ids:
            write_generic(f_out, 'i', tid)
        
        # Write token scores (all 0 since we don't have them easily)
        bkey = b"tokenizer.ggml.scores"
        write_generic(f_out, 'Q', len(bkey))
        f_out.write(bkey)
        write_generic(f_out, 'I', GGUF_VALUE_ARRAY)
        write_generic(f_out, 'I', GGUF_VALUE_FLOAT32)
        write_generic(f_out, 'Q', n_vocab)
        for _ in ids:
            write_generic(f_out, 'f', 0.0)
        
        # Write token text entries
        for tid in ids:
            text = token_texts[tid]
            key = f"tokenizer.ggml.token_{tid}.text"
            write_kv_str(f_out, key, text)
        
        new_hdr_end = f_out.tell()
        
        # Pad to alignment boundary (same as original)
        aligned = (new_hdr_end + alignment - 1) // alignment * alignment
        pad = aligned - new_hdr_end
        f_out.write(b'\x00' * pad)
        
        # Now write tensor info (recalculate offsets)
        data_pos = aligned
        for i, (name, ndim, dims, ggml_type, old_offset) in enumerate(tensor_infos):
            bname = name.encode('utf-8')
            write_generic(f_out, 'Q', len(bname))
            f_out.write(bname)
            write_generic(f_out, 'I', ndim)
            for d in dims:
                write_generic(f_out, 'Q', d)
            write_generic(f_out, 'I', ggml_type)
            write_generic(f_out, 'Q', data_pos)
            
            # Calculate tensor size
            if ggml_type == 0:
                elems = 1
                for d in dims:
                    elems *= d
                tensor_size = elems * 4
            elif ggml_type == 8:
                elems = 1
                for d in dims:
                    elems *= d
                tensor_size = (elems // 32) * 34
            else:
                tensor_size = 0
                for d in dims:
                    tensor_size += d * 4  # fallback
            
            aligned_size = (tensor_size + alignment - 1) // alignment * alignment
            data_pos += aligned_size
        
        # Now copy tensor data from original
        f_in.seek(hdr_end)
        remaining = os.fstat(f_in.fileno()).st_size - hdr_end
        shutil.copyfileobj(f_in, f_out, remaining)
        
        print(f"Written to {dst_path}")
        print(f"Header: {new_hdr_end} -> aligned {aligned}, tensor data at {aligned}")
        print(f"Total size: {os.fstat(f_out.fileno()).st_size}")

if __name__ == '__main__':
    main()
