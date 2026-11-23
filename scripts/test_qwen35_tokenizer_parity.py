#!/usr/bin/env python3
"""Reference parity test for the native Qwen3.5 GGUF tokenizer adapter."""
from __future__ import annotations
import json, os, pathlib, struct, subprocess, sys, tempfile, unicodedata, urllib.request

REVISION = "2b48083dfa97c3cbf0220cf1a5e6cffe0a511157"
BASE = f"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/{REVISION}"
WHITE_SPACE = {9,10,11,12,13,32,0x85,0xa0,0x1680,0x2028,0x2029,0x202f,0x205f,0x3000,*range(0x2000,0x200b)}

def download_json(url):
    req=urllib.request.Request(url,headers={"User-Agent":"ai-chat-test-tokenizer-parity/1"})
    with urllib.request.urlopen(req,timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))

def pack_string(value):
    raw=value.encode("utf-8")
    return struct.pack("<Q",len(raw))+raw

def meta_string(key,value):
    return pack_string(key)+struct.pack("<I",8)+pack_string(value)

def meta_u32(key,value):
    return pack_string(key)+struct.pack("<II",4,value)

def meta_bool(key,value):
    return pack_string(key)+struct.pack("<IB",7,1 if value else 0)

def meta_strings(key,values):
    return pack_string(key)+struct.pack("<IIQ",9,8,len(values))+b"".join(pack_string(v) for v in values)

def meta_i32(key,values):
    return pack_string(key)+struct.pack("<IIQ",9,5,len(values))+b"".join(struct.pack("<i",v) for v in values)

def build_gguf(data,config,destination):
    model=data["model"]
    if model.get("type")!="BPE": raise RuntimeError(f"Expected BPE, got {model.get('type')!r}")
    indexed={}
    for token,tid in model["vocab"].items():
        tid=int(tid)
        if tid in indexed and indexed[tid]!=token: raise RuntimeError(f"Duplicate vocabulary id {tid}")
        indexed[tid]=token
    token_types={}
    user_defined_contents=set()
    for item in data.get("added_tokens",[]):
        tid=int(item["id"]); token=item["content"]
        if tid in indexed and indexed[tid]!=token: raise RuntimeError(f"Added-token collision at {tid}")
        indexed[tid]=token
        token_types[tid]=3 if item.get("special",False) else 4
        if not item.get("special",False): user_defined_contents.add(token)
    if "<|repo_name|>" not in user_defined_contents:
        raise RuntimeError("Expected <|repo_name|> to be a non-special added token in the pinned reference")
    size=max(indexed)+1
    if len(indexed)!=size: raise RuntimeError(f"Vocabulary has missing IDs: {sorted(set(range(size))-set(indexed))[:8]}")
    tokens=[indexed[i] for i in range(size)]
    types=[token_types.get(i,1) for i in range(size)]
    merges=[]
    for merge in model.get("merges",[]):
        if isinstance(merge,list):
            if len(merge)!=2: raise RuntimeError(f"Bad merge {merge!r}")
            merges.append(merge[0]+" "+merge[1])
        else: merges.append(str(merge))
    def configured_id(key):
        token=config.get(key)
        if isinstance(token,dict): token=token.get("content")
        matches=[i for i,value in indexed.items() if value==token]
        if not matches: raise RuntimeError(f"{key} token {token!r} missing from vocabulary")
        return matches[0]
    metadata=[
        meta_string("general.architecture","qwen35"),
        meta_string("general.name","Qwen3.5-0.8B upstream tokenizer parity fixture"),
        meta_u32("general.alignment",32),
        meta_string("tokenizer.ggml.model","gpt2"),
        meta_string("tokenizer.ggml.pre","qwen35"),
        meta_strings("tokenizer.ggml.tokens",tokens),
        meta_strings("tokenizer.ggml.merges",merges),
        meta_i32("tokenizer.ggml.token_type",types),
        meta_u32("tokenizer.ggml.eos_token_id",configured_id("eos_token")),
        meta_u32("tokenizer.ggml.padding_token_id",configured_id("pad_token")),
        meta_bool("tokenizer.ggml.add_bos_token",False),
        meta_bool("tokenizer.ggml.add_eos_token",False),
        meta_bool("tokenizer.ggml.add_space_prefix",False),
        meta_bool("tokenizer.ggml.byte_fallback",False),
    ]
    header=b"GGUF"+struct.pack("<IQQ",3,1,len(metadata))
    tensor=(pack_string("tokenizer_parity_dummy")+struct.pack("<I",1)+struct.pack("<Q",1)+struct.pack("<I",0)+struct.pack("<Q",0))
    padding=(-(len(header)+sum(map(len,metadata))+len(tensor)))%32
    destination.write_bytes(header+b"".join(metadata)+tensor+b"\0"*padding+struct.pack("<f",0.0))

def unicode_kind(ch):
    cp=ord(ch)
    if cp in (10,13): return 3
    if cp in WHITE_SPACE: return 2
    category=unicodedata.category(ch)
    if category[0] in ("L","M"): return 0
    if category[0]=="N": return 1
    return 4

def write_categories(text,path):
    seen={ord(ch):unicode_kind(ch) for ch in text}
    path.write_text("".join(f"{cp} {kind}\n" for cp,kind in seen.items()),encoding="ascii")

def run_native(cli,gguf,text,categories):
    result=subprocess.run([cli,str(gguf),text,str(categories)],check=True,capture_output=True,text=True,encoding="utf-8")
    lines=result.stdout.splitlines()
    if len(lines)!=2: raise RuntimeError(f"Unexpected native output: {result.stdout!r}")
    ids=[int(v) for v in lines[0].split(",") if v]
    return ids,bytes.fromhex(lines[1]).decode("utf-8")

def main():
    if len(sys.argv)!=2:
        print("usage: test_qwen35_tokenizer_parity.py TOKENIZER_PARITY_CLI",file=sys.stderr); return 2
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("Install tokenizers==0.22.1 first") from error
    tokenizer_data=download_json(BASE+"/tokenizer.json")
    config=download_json(BASE+"/tokenizer_config.json")
    corpus=[
        "Hello, world!","  leading and trailing   ","1234567890",
        "we're testing Qwen3.5's tokenizer.","سلام", " سلام ", "سلام دنیا، این یک آزمایش است.",
        "می‌خواهم ۱۲۳ عدد و emoji 🧠🚀 را تست کنم.","e\u0301 cafe\u0301",
        "line one\r\nline two\n\t  "," punctuation!!! ... ???","   ",
        "emoji 🧑🏽‍💻 and symbols ©™","<|im_start|>assistant\nHello<|im_end|>",
        "<|im_start|>user\nسلام<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|im_start|>user\nسلام<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|repo_name|> my-repo",
        "中文、日本語、한국어 mixed with English 42","A\u030A and Å; NFC and decomposed forms.",
    ]
    cli=os.path.abspath(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="qwen35-tokenizer-parity-") as temp:
        root=pathlib.Path(temp)
        tokenizer_path=root/"tokenizer.json"
        tokenizer_path.write_text(json.dumps(tokenizer_data,ensure_ascii=False),encoding="utf-8")
        reference=Tokenizer.from_file(str(tokenizer_path))
        gguf=root/"qwen35-tokenizer.gguf"
        build_gguf(tokenizer_data,config,gguf)
        for index,text in enumerate(corpus):
            normalized=unicodedata.normalize("NFC",text)
            categories=root/f"categories-{index}.txt"
            write_categories(normalized,categories)
            expected=reference.encode(text,add_special_tokens=False).ids
            actual,decoded=run_native(cli,gguf,normalized,categories)
            if actual!=expected:
                raise AssertionError(f"Encode parity failed case {index}: {text!r}\nexpected {expected[:80]}\nactual {actual[:80]}")
            expected_decoded=reference.decode(expected,skip_special_tokens=True)
            if decoded!=expected_decoded:
                raise AssertionError(f"Decode parity failed case {index}: {text!r}\nexpected {expected_decoded!r}\nactual {decoded!r}")
    print(f"Qwen3.5 tokenizer reference parity passed: {len(corpus)} multilingual/Unicode cases.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
