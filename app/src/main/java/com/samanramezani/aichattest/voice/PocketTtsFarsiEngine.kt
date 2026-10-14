package com.samanramezani.aichattest.voice

import ai.onnxruntime.*
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import com.sentencepiece.Model
import com.sentencepiece.Scoring
import com.sentencepiece.SentencePieceAlgorithm
import java.io.Closeable
import java.io.File
import java.nio.*
import java.nio.file.Paths
import java.util.LinkedHashMap
import java.util.Random
import java.util.zip.ZipInputStream
import kotlin.math.*

/**
 * ONNX Runtime implementation of the upstream Pocket TTS Farsi v2 pipeline.
 * Model inference is offline and CPU-backed, like the upstream reference script.
 */
internal class PocketTtsFarsiEngine(private val dir: File) : java.io.Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val opts = OrtSession.SessionOptions()
    private val flow: OrtSession
    private val enc: OrtSession
    private val dec: OrtSession
    private val g2p: PocketTtsFarsiG2p
    private val sp: Model
    private val algo = SentencePieceAlgorithm(true, Scoring.HIGHEST_SCORE)
    private val c = org.json.JSONObject(File(dir, "manifest.json").readText()).getJSONObject("constants")
    private val ld = c.getInt("ldim"); private val dim = c.getInt("dim")
    private val layers = c.getInt("layers"); private val heads = c.getInt("heads")
    private val hd = c.getInt("dim_per_head"); private val cap = c.getInt("cache_capacity")
    private val sr = c.getInt("sample_rate"); private val steps = c.getInt("mimi_steps_per_latent")
    private val temp = sqrt(c.getDouble("temp")).toFloat()
    private val tps = c.getDouble("tokens_per_second_estimate")
    private val pad = c.getDouble("gen_seconds_padding")
    private val fps = c.getDouble("frame_rate")
    private val weights = npz(File(dir, "weights.npz"))
    private val lut = weights.getValue("lut_weight").v
    private val lutShape = weights.getValue("lut_weight").shape
    private val proj = weights.getValue("speaker_proj").v
    private val projShape = weights.getValue("speaker_proj").shape
    private val bos = weights.getValue("bos_before_voice").v
    private val std = weights.getValue("emb_std").v
    private val mean = weights.getValue("emb_mean").v
    private val decInputs: List<String>
    private val decOutputs: List<String>
    private val decoderInitial: List<A>
    private val cacheSlots: List<Int>
    private val smallSlots: List<Int>
    private val smallOutputs: List<Int>
    private val kvOutputs: List<Int>
    private val rng = Random()

    init {
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        opts.setIntraOpNumThreads(2); opts.setInterOpNumThreads(1)
        flow = env.createSession(File(dir, "flow_lm_step.onnx").absolutePath, opts)
        enc = env.createSession(File(dir, "mimi_encoder.onnx").absolutePath, opts)
        dec = env.createSession(File(dir, "mimi_decoder_step_kv.onnx").absolutePath, opts)
        g2p = PocketTtsFarsiG2p(dir)
        sp = Model.parseFrom(Paths.get(File(dir, "tokenizer_ph.model").absolutePath))
        val states = npz(File(dir, "decode_state_init.npz"))
        decInputs = dec.inputNames.toList(); decOutputs = dec.outputNames.toList()
        val expectedDecoderInputs = setOf("latent") + (0 until states.size).map { "s$it" }
        require(decInputs.toSet() == expectedDecoderInputs) {
            "Mimi decoder input names mismatch: expected=$expectedDecoderInputs actual=${decInputs.toSet()}"
        }
        decoderInitial = states.values.map { A(it.name, it.shape, it.v, it.name.endsWith(".cache"), it.dtype, it.lv) }
        cacheSlots = decoderInitial.indices.filter { decoderInitial[it].cache }
        smallSlots = decoderInitial.indices.filterNot { it in cacheSlots }
        // ONNX Runtime exposes input/output names as sets; never rely on their iteration order.
        // State outputs are numbered by their original decoder-state slot, while KV
        // outputs are numbered by cache slot.
        smallOutputs = decOutputs.indices
            .filter { decOutputs[it].endsWith("o") && decOutputs[it] != "audio" }
            .sortedBy { decOutputs[it].removePrefix("s").removeSuffix("o").toIntOrNull() ?: Int.MAX_VALUE }
        kvOutputs = decOutputs.indices
            .filter { decOutputs[it].startsWith("kv") }
            .sortedBy { decOutputs[it].removePrefix("kv").toIntOrNull() ?: Int.MAX_VALUE }
        require(smallOutputs.size == smallSlots.size && kvOutputs.size == cacheSlots.size) {
            "Mimi state output layout does not match decode_state_init.npz"
        }
    }

    /** Returns mono float PCM at 24 kHz; caller owns playback and cancellation. */
    @Synchronized fun synthesize(text: String, cancelled: () -> Boolean = { false }): FloatArray {
        require(text.isNotBlank()) { "POCKET-TTS-001: متن خالی است." }
        val parts = mutableListOf<FloatArray>()
        val sentences = text.replace('\r',' ').split(Regex("(?<=[.!?؟؛])\\s+|[\\n]+")).map(String::trim).filter(String::isNotBlank)
        for ((si, sentence) in sentences.withIndex()) {
            check(!cancelled()) { "Pocket TTS cancelled" }
            val phonemes = g2p.phonemise(sentence)
            val words = phonemes.split(Regex("\\s+")).filter(String::isNotBlank)
            var chunk = mutableListOf<String>()
            val chunks = mutableListOf<String>()
            for (word in words) {
                val candidate = (chunk + word).joinToString(" ")
                if (chunk.isNotEmpty() && sp.encodeNormalized(candidate, algo).size > 18) {
                    chunks += chunk.joinToString(" "); chunk = mutableListOf(word)
                } else chunk += word
            }
            if (chunk.isNotEmpty()) chunks += chunk.joinToString(" ")
            for ((ci, phrase) in chunks.withIndex()) {
                val latents = generate(phrase, cancelled)
                val audio = decode(latents)
                if (audio.isNotEmpty()) parts += audio
                if (ci < chunks.lastIndex) parts += FloatArray((sr * .16).toInt())
            }
            if (si < sentences.lastIndex) parts += FloatArray((sr * .32).toInt())
        }
        return concat(parts)
    }

    private fun generate(phrase: String, cancelled: () -> Boolean): List<FloatArray> {
        val tokens = sp.encodeNormalized(phrase, algo)
        require(tokens.isNotEmpty()) { "Pocket tokenizer returned no tokens" }
        require(lutShape.size == 2 && lutShape[1] == dim.toLong()) { "Bad LUT shape ${lutShape.contentToString()}" }
        val text = FloatArray(tokens.size * dim)
        tokens.forEachIndexed { i, id ->
            require(id >= 0 && id < lutShape[0]) { "Token ID outside embedding LUT: $id" }
            System.arraycopy(lut, id * dim, text, i * dim, dim)
        }
        val (voiceCache, voiceOffset) = voiceCache()
        val cache = voiceCache.copyOf()
        var offset = voiceOffset
        var latent: FloatArray
        var eos = false
        val initial = flowStep(FloatArray(0), longArrayOf(1,0,ld.toLong()), text,
            longArrayOf(1,tokens.size.toLong(),dim.toLong()), offset, noise(), cache)
        latent = initial.latent; offset = initial.offset; initial.cache.copyInto(cache)
        val result = mutableListOf(latent)
        val maxFrames = ceil((tokens.size / tps + pad) * fps).toInt().coerceIn(1,512)
        var eosAt = -1
        val afterEos = (if (phrase.split(Regex("\\s+")).size <= 4) 3 else 1) + 2
        for (i in 0 until maxFrames) {
            check(!cancelled()) { "Pocket TTS cancelled" }
            val next = flowStep(latent, longArrayOf(1,1,ld.toLong()), FloatArray(0),
                longArrayOf(1,0,dim.toLong()), offset, noise(), cache)
            latent = next.latent; offset = next.offset; next.cache.copyInto(cache); eos = next.eos
            if (eos && eosAt < 0) eosAt = i
            if (eosAt >= 0 && i >= eosAt + afterEos) break
            result += latent
        }
        return result
    }

    private data class F(val latent: FloatArray, val offset: Int, val cache: FloatArray, val eos: Boolean)
    private fun flowStep(seq: FloatArray, seqShape: LongArray, txt: FloatArray, txtShape: LongArray,
                         offset: Int, noise: FloatArray, cache: FloatArray): F {
        val a = tensor(seq,seqShape); val b = tensor(txt,txtShape); val n = tensor(noise,longArrayOf(1,ld.toLong()))
        val o = OnnxTensor.createTensor(env,LongBuffer.wrap(longArrayOf(offset.toLong())),longArrayOf())
        val k = tensor(cache,longArrayOf(layers.toLong(),2,cap.toLong(),heads.toLong(),hd.toLong()))
        try {
            flow.run(mapOf("sequence" to a,"text_emb" to b,"offset" to o,"noise" to n,"cache" to k)).use { r ->
                val names=flow.outputNames.toList()
                val li=names.indexOfFirst { it.contains("latent",true) }.let { if(it<0) 0 else it }
                val ei=names.indexOfFirst { it.contains("eos",true) }.let { if(it<0) 1 else it }
                val ki=names.indexOfFirst { it.contains("kv",true)||it.contains("cache",true) }.let { if(it<0) 2 else it }
                val lat=floats(r[li] as? OnnxTensor ?: error("Pocket flow latent output is not an ONNX tensor: ${names[li]}"))
                val eos=floats(r[ei] as? OnnxTensor ?: error("Pocket flow EOS output is not an ONNX tensor: ${names[ei]}")).firstOrNull()?.let { it>.5f } ?: false
                val kt=r[ki] as? OnnxTensor ?: error("Pocket flow KV output is not an ONNX tensor: ${names[ki]}"); val kv=floats(kt)
                val count=kv.size/(layers*2*heads*hd)
                require(count>0 && offset+count<=cap) { "Flow KV cache overflow: offset=$offset count=$count cap=$cap" }
                val updated=cache.copyOf()
                for(l in 0 until layers) for(kvKind in 0..1) for(s in 0 until count)
                    for(h in 0 until heads) for(d in 0 until hd) {
                        val src=((((l*2+kvKind)*count+s)*heads+h)*hd+d)
                        val dst=((((l*2+kvKind)*cap+offset+s)*heads+h)*hd+d)
                        updated[dst]=kv[src]
                    }
                return F(lat,offset+count,updated,eos)
            }
        } finally { a.close();b.close();n.close();o.close();k.close() }
    }

    private fun voiceCache(): Pair<FloatArray,Int> {
        val audio=readWav(File(dir,"male_hello.wav"))
        val at=tensor(audio,longArrayOf(1,1,audio.size.toLong()))
        val latent=try { enc.run(mapOf("audio" to at)).use { floats(it[0] as? OnnxTensor ?: error("Mimi encoder output is not an ONNX tensor: ${enc.outputNames.firstOrNull()}")) } } finally { at.close() }
        val count=latent.size/ld
        require(count>0 && projShape.size==2 && projShape[1]==ld.toLong()) { "Bad speaker conditioning tensors" }
        val text=FloatArray((count+1)*dim)
        System.arraycopy(bos,0,text,0,min(dim,bos.size))
        for(t in 0 until count) for(d in 0 until dim) {
            var sum=0f; for(k in 0 until ld) sum+=latent[t*ld+k]*proj[d*ld+k]
            text[(t+1)*dim+d]=sum
        }
        val cache=FloatArray(layers*2*cap*heads*hd)
        val a=tensor(FloatArray(0),longArrayOf(1,0,ld.toLong()))
        val b=tensor(text,longArrayOf(1,count+1L,dim.toLong()))
        val n=tensor(FloatArray(ld),longArrayOf(1,ld.toLong()))
        val o=OnnxTensor.createTensor(env,LongBuffer.wrap(longArrayOf(0)),longArrayOf())
        val k=tensor(cache,longArrayOf(layers.toLong(),2,cap.toLong(),heads.toLong(),hd.toLong()))
        try {
            flow.run(mapOf("sequence" to a,"text_emb" to b,"offset" to o,"noise" to n,"cache" to k)).use { r ->
                val names=flow.outputNames.toList()
                val ix=names.indexOfFirst { it.contains("kv",true)||it.contains("cache",true) }.let { if(it<0) 2 else it }
                val kv=floats(r[ix] as? OnnxTensor ?: error("Pocket voice-cache KV output is not an ONNX tensor: ${flow.outputNames.elementAtOrNull(ix)}"))
                val countSteps=kv.size/(layers*2*heads*hd)
                require(countSteps<=cap) { "Voice prompt exceeds cache capacity" }
                for(l in 0 until layers) for(kind in 0..1) for(s in 0 until countSteps)
                    for(h in 0 until heads) for(d in 0 until hd) {
                        val src=((((l*2+kind)*countSteps+s)*heads+h)*hd+d)
                        val dst=((((l*2+kind)*cap+s)*heads+h)*hd+d)
                        cache[dst]=kv[src]
                    }
                return cache to countSteps
            }
        } finally { a.close();b.close();n.close();o.close();k.close() }
    }

    private fun decode(latents: List<FloatArray>): FloatArray {
        val states=decoderInitial.map { A(it.name,it.shape,it.v.copyOf(),it.cache,it.dtype,it.lv?.copyOf()) }.toMutableList()
        val audio=mutableListOf<FloatArray>(); var offset=0
        for(latent in latents) {
            val un=FloatArray(ld) { i -> latent.getOrElse(i){0f}*std.getOrElse(i){1f}+mean.getOrElse(i){0f} }
            val lt=tensor(un,longArrayOf(1,1,ld.toLong()))
            val stateTensors=mutableListOf<OnnxTensor>()
            try {
                // ONNX Runtime does not guarantee inputNames iteration order. Bind by
                // the exported graph names so a float latent can never be fed into a
                // BOOL/INT decoder-state input merely because it appeared first.
                val feeds=LinkedHashMap<String,OnnxTensor>(); feeds["latent"]=lt
                states.forEachIndexed { i,s ->
                    val inputName = "s$i"
                    tensor(s).also { stateTensors+=it; feeds[inputName]=it }
                }
                dec.run(feeds).use { r ->
                    val ai=decOutputs.indexOf("audio").let { if(it<0) 0 else it }
                    audio += floats(r[ai] as? OnnxTensor ?: error("Mimi audio output is not an ONNX tensor: ${decOutputs.getOrNull(ai)}"))
                    smallOutputs.forEachIndexed { j,oi ->
                        val slot=smallSlots[j]; val ot=r[oi] as? OnnxTensor ?: error("Mimi state output is not an ONNX tensor: ${decOutputs.getOrNull(oi)}")
                        states[slot]=stateFromTensor(states[slot].name, ot, false)
                    }
                    kvOutputs.forEachIndexed { j,oi ->
                        val slot=cacheSlots[j]; val ot=r[oi] as? OnnxTensor ?: error("Mimi state output is not an ONNX tensor: ${decOutputs.getOrNull(oi)}"); val out=floats(ot)
                        val state=states[slot]; val per=state.v.size/state.shape[2].toInt()
                        val n=min(steps,out.size/per)
                        require(offset+n<=state.shape[2]) { "Mimi KV cache overflow" }
                        System.arraycopy(out,0,state.v,offset*per,n*per)
                    }
                }
            } finally { lt.close();stateTensors.forEach { it.close() } }
            offset += steps
        }
        return concat(audio)
    }

    private data class A(val name:String,val shape:LongArray,val v:FloatArray,val cache:Boolean,val dtype:String = "<f4",val lv:LongArray? = null)
    private fun npz(file:File):LinkedHashMap<String,A> {
        val out=LinkedHashMap<String,A>()
        ZipInputStream(file.inputStream().buffered()).use { z ->
            while(true) { val e=z.nextEntry?:break
                if(!e.isDirectory && e.name.endsWith(".npy")) {
                    val bytes=z.readBytes(); val major=bytes[6].toInt() and 255
                    val hlen=if(major>=2) ByteBuffer.wrap(bytes,8,4).order(ByteOrder.LITTLE_ENDIAN).int else ByteBuffer.wrap(bytes,8,2).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 65535
                    val hs=if(major>=2)12 else 10; val header=String(bytes,hs,hlen,Charsets.US_ASCII)
                    val dtype=Regex("""['"]descr['"]\s*:\s*['"]([^'"]+)""").find(header)?.groupValues?.get(1)?:error("NPY dtype missing")
                    val isBool = dtype == "|b1" || dtype == "?b1"
                    val isInt64 = dtype.endsWith("i8")
                    require(dtype.endsWith("f4") || dtype.endsWith("f8") || isBool || isInt64) {
                        "Unsupported NPZ dtype $dtype in ${file.name}:${e.name}"
                    }
                    val st=Regex("""['"]shape['"]\s*:\s*\(([^)]*)""").find(header)?.groupValues?.get(1).orEmpty()
                    val shape=st.split(',').mapNotNull { it.trim().toLongOrNull() }.toLongArray()
                    val n=if(shape.isEmpty())1L else shape.fold(1L){a,b->Math.multiplyExact(a,b)}
                    val bytesPer=when { isBool -> 1; dtype.endsWith("f8") || isInt64 -> 8; else -> 4 }
                    val pos=hs+hlen
                    require(n <= Int.MAX_VALUE && pos.toLong()+n*bytesPer<=bytes.size) {
                        "NPY payload truncated or too large: ${e.name} shape=${shape.contentToString()} dtype=$dtype"
                    }
                    val byteOrder = if (dtype.startsWith(">")) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
                    val bb=ByteBuffer.wrap(bytes,pos,(n*bytesPer).toInt()).order(byteOrder)
                    val values = if (isInt64) FloatArray(0) else FloatArray(n.toInt()) {
                        when {
                            isBool -> if (bb.get().toInt() != 0) 1f else 0f
                            bytesPer==4 -> bb.float
                            else -> bb.double.toFloat()
                        }
                    }
                    val longs = if (isInt64) LongArray(n.toInt()) { bb.long } else null
                    val name=e.name.removeSuffix(".npy")
                    out[name]=A(name,shape,values,name.endsWith(".cache"),dtype,longs)
                }
                z.closeEntry()
            }
        }
        require(out.isNotEmpty()) { "Invalid/empty NPZ: ${file.name}" }; return out
    }

    private fun readWav(file:File):FloatArray {
        val b=file.readBytes(); require(b.size>=44&&String(b,0,4)=="RIFF"&&String(b,8,4)=="WAVE") { "Invalid reference WAV" }
        val bb=ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN); var p=12; var fmt=1; var ch=1; var rate=sr; var bits=16; var data=-1; var len=0
        while(p+8<=b.size) { val id=String(b,p,4); val n=(bb.getInt(p+4)).coerceAtLeast(0); val start=p+8
            if(id=="fmt "&&start+16<=b.size){fmt=bb.getShort(start).toInt() and 65535;ch=bb.getShort(start+2).toInt() and 65535;rate=bb.getInt(start+4);bits=bb.getShort(start+14).toInt() and 65535}
            if(id=="data"){data=start;len=min(n,b.size-start);break};p=start+n+(n and 1)
        }
        require(data>=0&&ch>0&&(fmt==1&&bits==16||fmt==3&&bits==32)) { "Unsupported reference WAV format" }
        val bps=bits/8;val frames=len/(bps*ch);val mono=FloatArray(frames)
        for(i in 0 until frames){var s=0f;for(c in 0 until ch){val at=data+(i*ch+c)*bps;s+=if(fmt==1)bb.getShort(at)/32768f else bb.getFloat(at)};mono[i]=(s/ch).coerceIn(-1f,1f)}
        if(rate==sr)return mono
        val n=(frames.toDouble()*sr/rate).toInt()
        return FloatArray(n){i->val x=i.toDouble()*rate/sr;val a=min(frames-1,x.toInt().coerceAtLeast(0));val d=min(frames-1,a+1);val f=(x-a).toFloat();mono[a]*(1-f)+mono[d]*f}
    }

    private fun noise()=FloatArray(ld){val u=max(1e-7,rng.nextDouble());(sqrt(-2*ln(u))*cos(2*PI*rng.nextDouble())*temp).toFloat()}
    private fun tensor(v:FloatArray,s:LongArray)=OnnxTensor.createTensor(env,FloatBuffer.wrap(v),s)
    private fun tensor(state:A):OnnxTensor = when {
        state.dtype.endsWith("i8") || state.lv != null -> {
            val values = state.lv ?: error("Integer state ${state.name} has no integer payload")
            val buffer = ByteBuffer.allocate(values.size * 8).order(ByteOrder.nativeOrder()).asLongBuffer()
            buffer.put(values); buffer.flip()
            OnnxTensor.createTensor(env, buffer, state.shape)
        }
        state.dtype == "|b1" || state.dtype == "?b1" -> {
            // ONNX BOOL tensors are byte-backed in the Java API; feeding the
            // NPZ values as FloatBuffer makes ORT reject them at decoder input.
            val buffer = ByteBuffer.allocateDirect(state.v.size).order(ByteOrder.nativeOrder())
            state.v.forEach { buffer.put(if (it != 0f) 1.toByte() else 0.toByte()) }
            buffer.flip()
            OnnxTensor.createTensor(env, buffer, state.shape, OnnxJavaType.BOOL)
        }
        else -> tensor(state.v, state.shape)
    }
    private fun floats(t:OnnxTensor):FloatArray {
        val b=t.floatBuffer.duplicate()
        return FloatArray(b.remaining()).also{b.get(it)}
    }
    private fun stateFromTensor(name:String,t:OnnxTensor,cache:Boolean):A {
        val shape=t.info.shape
        return when (t.info.type) {
            OnnxJavaType.INT64 -> {
                val b=t.longBuffer.duplicate()
                val values=LongArray(b.remaining()).also { b.get(it) }
                A(name,shape,FloatArray(0),cache,"<i8",values)
            }
            OnnxJavaType.BOOL -> {
                val b=t.byteBuffer
                val values=FloatArray(b.remaining()) { if (b.get().toInt() != 0) 1f else 0f }
                A(name,shape,values,cache,"|b1",null)
            }
            else -> A(name,shape,floats(t),cache,"<f4",null)
        }
    }
    private fun concat(a:List<FloatArray>):FloatArray { val out=FloatArray(a.sumOf{it.size});var p=0;for(x in a){x.copyInto(out,p);p+=x.size};return out }
    override fun close(){runCatching{g2p.close()};runCatching{flow.close()};runCatching{enc.close()};runCatching{dec.close()};opts.close()}
}
