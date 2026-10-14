#!/usr/bin/env python3
"""CI-only contract test for the Pocket TTS Farsi v2 ONNX bundle."""
import json, os, urllib.request, zipfile
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT=Path(os.environ.get("POCKET_TTS_TEST_DIR","/tmp/pocket-tts-contract"))
BASE="https://huggingface.co/Nimaone/pocket-tts-farsi-v2-onnx/resolve/main/"
MIN={"manifest.json":100,"weights.npz":1_000_000,"decode_state_init.npz":1_000_000,
"flow_lm_step.onnx":100_000,"flow_lm_step.onnx.data":100_000_000,
"g2p_encoder.onnx":1_000_000,"g2p_decoder.onnx":100_000,"g2p_decoder.onnx.data":1_000_000,
"mimi_encoder.onnx":100_000,"mimi_encoder.onnx.data":10_000_000,
"mimi_decoder_step_kv.onnx":100_000,"mimi_decoder_step_kv.onnx.data":10_000_000}
def need(ok,msg):
 if not ok: raise AssertionError(msg)
def getfile(name):
 p=ROOT/name
 if p.exists() and p.stat().st_size>=MIN[name]: return p
 ROOT.mkdir(parents=True,exist_ok=True)
 req=urllib.request.Request(BASE+name+"?download=true",headers={"User-Agent":"ai-chat-test-ci/1.0"})
 tmp=p.with_suffix(p.suffix+".part")
 print("Downloading reference artifact:",name,flush=True)
 with urllib.request.urlopen(req,timeout=180) as src,tmp.open("wb") as out:
  while True:
   b=src.read(1024*1024)
   if not b: break
   out.write(b)
 need(tmp.stat().st_size>=MIN[name],f"{name} download incomplete: {tmp.stat().st_size} bytes")
 tmp.replace(p); return p
def typ(node,want,graph): need(node.type==want,f"{graph}.{node.name}: expected {want}, got {node.type}")
def rank(node,want,graph): need(len(node.shape)==want,f"{graph}.{node.name}: expected rank {want}, got {node.shape}")
def main():
 for n in MIN: getfile(n)
 c=json.loads((ROOT/"manifest.json").read_text())["constants"]
 for k,v in {"sample_rate":24000,"ldim":32,"dim":1024,"layers":6,"heads":16,"dim_per_head":64}.items():
  need(c.get(k)==v,f"manifest.constants.{k}: expected {v}, got {c.get(k)}")
 cap=c.get("cache_capacity"); need(isinstance(cap,int) and cap>=256,f"invalid cache_capacity {cap}")
 with zipfile.ZipFile(ROOT/"weights.npz") as z: arrays=set(z.namelist())
 required={"lut_weight.npy","speaker_proj.npy","bos_before_voice.npy","emb_std.npy","emb_mean.npy"}
 need(required<=arrays,f"weights.npz missing {sorted(required-arrays)}")
 with np.load(ROOT/"decode_state_init.npz",allow_pickle=False) as z:
  states=set(z.files); need(bool(states),"decode_state_init.npz contains no decoder states")
  for n in z.files: need(z[n].size>0,f"empty decoder state {n}")
 opt=ort.SessionOptions(); opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
 opt.intra_op_num_threads=2; opt.inter_op_num_threads=1
 gs={}
 for n in ("flow_lm_step.onnx","g2p_encoder.onnx","g2p_decoder.onnx","mimi_encoder.onnx","mimi_decoder_step_kv.onnx"):
  print("Opening ONNX Runtime graph:",n,flush=True)
  gs[n]=ort.InferenceSession(str(ROOT/n),sess_options=opt,providers=["CPUExecutionProvider"])
 f=gs["flow_lm_step.onnx"]; fi={x.name:x for x in f.get_inputs()}; fo={x.name:x for x in f.get_outputs()}
 need(set(fi)=={"sequence","text_emb","offset","noise","cache"},f"Flow inputs mismatch {sorted(fi)}")
 need(len(fo)==3 and any("latent" in n.lower() for n in fo) and any("eos" in n.lower() for n in fo) and any(("kv" in n.lower() or "cache" in n.lower()) for n in fo),f"Flow outputs mismatch {sorted(fo)}")
 for n,r in (("sequence",3),("text_emb",3),("offset",0),("noise",2),("cache",5)): rank(fi[n],r,"Flow")
 for n in ("sequence","text_emb","noise","cache"): typ(fi[n],"tensor(float)","Flow")
 typ(fi["offset"],"tensor(int64)","Flow")
 need(fi["sequence"].shape[-1]==c["ldim"],"Flow sequence latent dimension mismatch")
 need(fi["text_emb"].shape[-1]==c["dim"],"Flow text embedding dimension mismatch")
 need(fi["noise"].shape[-1]==c["ldim"],"Flow noise latent dimension mismatch")
 need(fi["cache"].shape==[c["layers"],2,cap,c["heads"],c["dim_per_head"]],f"Flow cache shape mismatch {fi['cache'].shape}")
 for n,x in fo.items():
  typ(x,"tensor(float)","Flow")
  if "latent" in n.lower(): rank(x,2,"Flow"); need(x.shape[-1]==c["ldim"],f"Flow latent shape mismatch {x.shape}")
 e=gs["g2p_encoder.onnx"]; ei={x.name:x for x in e.get_inputs()}; eo={x.name:x for x in e.get_outputs()}
 need(set(ei)=={"input_ids"} and bool(eo),f"G2P encoder signature mismatch: {sorted(ei)}, {sorted(eo)}")
 typ(ei["input_ids"],"tensor(int64)","G2P encoder"); rank(ei["input_ids"],2,"G2P encoder")
 typ(next(iter(eo.values())),"tensor(float)","G2P encoder"); rank(next(iter(eo.values())),3,"G2P encoder")
 d=gs["g2p_decoder.onnx"]; di={x.name:x for x in d.get_inputs()}; do={x.name:x for x in d.get_outputs()}
 need(set(di)=={"decoder_input_ids","encoder_hidden"} and bool(do),f"G2P decoder signature mismatch: {sorted(di)}, {sorted(do)}")
 typ(di["decoder_input_ids"],"tensor(int64)","G2P decoder"); rank(di["decoder_input_ids"],2,"G2P decoder")
 typ(di["encoder_hidden"],"tensor(float)","G2P decoder"); rank(di["encoder_hidden"],3,"G2P decoder")
 typ(next(iter(do.values())),"tensor(float)","G2P decoder"); rank(next(iter(do.values())),3,"G2P decoder")
 me=gs["mimi_encoder.onnx"]; mei={x.name:x for x in me.get_inputs()}; meo={x.name:x for x in me.get_outputs()}
 need(set(mei)=={"audio"} and bool(meo),f"Mimi encoder signature mismatch: {sorted(mei)}, {sorted(meo)}")
 typ(mei["audio"],"tensor(float)","Mimi encoder"); rank(mei["audio"],3,"Mimi encoder")
 need(mei["audio"].shape[:2]==[1,1],"Mimi encoder audio must be [1,1,T]")
 latent=next(iter(meo.values())); typ(latent,"tensor(float)","Mimi encoder"); rank(latent,3,"Mimi encoder")
 need(latent.shape[-1]==c["ldim"],f"Mimi latent dimension mismatch {latent.shape}")
 md=gs["mimi_decoder_step_kv.onnx"]; mdi={x.name:x for x in md.get_inputs()}; mdo={x.name:x for x in md.get_outputs()}
 expected=states|{"latent"}
 need(set(mdi)==expected,f"Mimi inputs/state mismatch; missing={sorted(expected-set(mdi))}; unexpected={sorted(set(mdi)-expected)}")
 need("audio" in mdo,f"Mimi audio output missing: {sorted(mdo)}")
 caches=sum(n.endswith(".cache") for n in states)
 need(sum(n.startswith("kv") for n in mdo)==caches,"Mimi KV output count differs from cache states")
 need(sum(n.endswith("o") and n!="audio" for n in mdo)==len(states)-caches,"Mimi small-state output count differs from state inputs")
 typ(mdi["latent"],"tensor(float)","Mimi decoder"); rank(mdi["latent"],3,"Mimi decoder")
 need(mdi["latent"].shape[:2]==[1,1] and mdi["latent"].shape[-1]==c["ldim"],f"Mimi latent shape mismatch {mdi['latent'].shape}")
 typ(mdo["audio"],"tensor(float)","Mimi decoder"); rank(mdo["audio"],3,"Mimi decoder")
 need(mdo["audio"].shape[:2]==[1,1],f"Mimi audio output shape mismatch {mdo['audio'].shape}")
 print("POCKET-TTS-CONTRACT-PASS: five ONNX graphs loaded; signatures, dtypes, ranks, dimensions, weights and decoder states validated.",flush=True)
if __name__=="__main__": main()
