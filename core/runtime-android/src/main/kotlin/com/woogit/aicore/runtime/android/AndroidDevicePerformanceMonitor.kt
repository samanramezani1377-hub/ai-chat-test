package com.woogit.aicore.runtime.android

import android.os.Process
import android.os.SystemClock
import java.io.File
import java.util.Locale

/**
 * Samples Linux/Android procfs and sysfs while a native inference is running.
 * Unreadable kernel nodes stay unavailable; this collector never fabricates values.
 */
internal class AndroidDevicePerformanceMonitor private constructor() {
    private data class CpuCounters(val total: Long, val idle: Long)
    private data class CpuSample(
        val counters: Map<Int, CpuCounters>,
        val frequenciesKhz: Map<Int, Long>,
        val processCpuMs: Long,
        val elapsedMs: Long,
    )

    private val lock = Any()
    @Volatile private var running = true
    private val startedAt = SystemClock.elapsedRealtime()
    private val samples = mutableListOf<CpuSample>()
    private val thermalSamples = mutableMapOf<String, MutableList<Double>>()
    private var peakRssKb: Long? = null
    private var lastMemoryInfo: Map<String, Long> = emptyMap()
    private var lastPsi: String? = null
    private val sampler = Thread({
        while (running) {
            sampleOnce()
            try {
                Thread.sleep(SAMPLE_INTERVAL_MS)
            } catch (_: InterruptedException) {
                break
            }
        }
        sampleOnce()
    }, "ai-chat-device-perf").apply { isDaemon = true; start() }

    fun stopAndReport(): String {
        running = false
        sampler.interrupt()
        try { sampler.join(750) } catch (_: InterruptedException) { Thread.currentThread().interrupt() }
        val endedAt = SystemClock.elapsedRealtime()
        val captured = synchronized(lock) {
            Triple(samples.toList(), thermalSamples.mapValues { it.value.toList() }, lastMemoryInfo.toMap()) to (peakRssKb to lastPsi)
        }
        val cpuSamples = captured.first.first
        val thermal = captured.first.second
        val memory = captured.first.third
        val extra = captured.second
        return buildString {
            appendLine("===== DEVICE PERFORMANCE / THERMAL =====")
            appendLine("Sampling Duration: ${(endedAt - startedAt).coerceAtLeast(0)} ms")
            appendLine("Sampling Interval: ${SAMPLE_INTERVAL_MS} ms")
            appendLine("Samples Collected: ${cpuSamples.size}")
            val coreIds = cpuSamples.flatMap { it.counters.keys }.distinct().sorted()
            appendLine("Detected CPU Cores: ${coreIds.size.takeIf { it > 0 } ?: "Unavailable"}")
            val processCpu = cpuSamples.zipWithNext().mapNotNull { (a, b) ->
                val elapsed = b.elapsedMs - a.elapsedMs
                if (elapsed <= 0) null else ((b.processCpuMs - a.processCpuMs).coerceAtLeast(0) * 100.0 / elapsed)
            }
            appendLine("Process CPU Average: ${processCpu.takeIf { it.isNotEmpty() }?.average()?.let { "%.1f%%".format(Locale.US, it) } ?: "Unavailable"}")
            appendLine("Process CPU Peak: ${processCpu.maxOrNull()?.let { "%.1f%%".format(Locale.US, it) } ?: "Unavailable"}")
            appendLine("CPU Usage Per Core (average across sampled intervals):")
            if (coreIds.isEmpty()) appendLine("  Unavailable: /proc/stat could not be read")
            coreIds.forEach { core ->
                val usage = cpuSamples.zipWithNext().mapNotNull { (a, b) ->
                    val before = a.counters[core] ?: return@mapNotNull null
                    val after = b.counters[core] ?: return@mapNotNull null
                    val total = after.total - before.total
                    val idle = after.idle - before.idle
                    if (total <= 0) null else ((total - idle).coerceAtLeast(0) * 100.0 / total).coerceIn(0.0, 100.0)
                }
                val frequencies = cpuSamples.mapNotNull { it.frequenciesKhz[core] }.map { it / 1000.0 }
                appendLine("  cpu$core usageAvg=${usage.takeIf { it.isNotEmpty() }?.average()?.let { "%.1f%%".format(Locale.US, it) } ?: "Unavailable"} freqMin=${frequencies.minOrNull()?.let { "%.0f MHz".format(Locale.US, it) } ?: "Unavailable"} freqAvg=${frequencies.takeIf { it.isNotEmpty() }?.average()?.let { "%.0f MHz".format(Locale.US, it) } ?: "Unavailable"} freqMax=${frequencies.maxOrNull()?.let { "%.0f MHz".format(Locale.US, it) } ?: "Unavailable"}")
            }
            appendLine()
            appendLine("Thermal Sensors (readable sysfs zones):")
            if (thermal.isEmpty()) appendLine("  Unavailable: thermal-zone sysfs access denied or absent")
            thermal.toSortedMap().forEach { (name, values) ->
                appendLine("  $name last=${values.lastOrNull()?.let { "%.1f C".format(Locale.US, it) } ?: "Unavailable"} max=${values.maxOrNull()?.let { "%.1f C".format(Locale.US, it) } ?: "Unavailable"}")
            }
            appendLine("Battery Temperature: ${thermal["battery"]?.lastOrNull()?.let { "%.1f C".format(Locale.US, it) } ?: "Unavailable"}")
            appendLine()
            appendLine("Memory (latest readable snapshot):")
            appendLine("  MemTotal: ${memory["MemTotal"]?.let { "$it kB" } ?: "Unavailable"}")
            appendLine("  MemAvailable: ${memory["MemAvailable"]?.let { "$it kB" } ?: "Unavailable"}")
            appendLine("  SwapTotal: ${memory["SwapTotal"]?.let { "$it kB" } ?: "Unavailable"}")
            appendLine("  SwapFree: ${memory["SwapFree"]?.let { "$it kB" } ?: "Unavailable"}")
            appendLine("  App RSS (sampled peak): ${extra.first?.let { "$it kB" } ?: "Unavailable"}")
            appendLine("  App VmHWM: ${memory["VmHWM"]?.let { "$it kB" } ?: "Unavailable"}")
            appendLine("  App Threads: ${memory["Threads"]?.toString() ?: "Unavailable"}")
            appendLine("  Memory PSI: ${extra.second ?: "Unavailable (kernel node inaccessible or unsupported)"}")
            appendLine("Note: CPU frequencies and thermal zones are reported only when Android exposes their sysfs nodes to this app.")
        }
    }

    private fun sampleOnce() {
        val elapsed = SystemClock.elapsedRealtime()
        val counters = readCpuCounters()
        val frequencies = counters.keys.mapNotNull { core ->
            val currentKhz = readLong("/sys/devices/system/cpu/cpu$core/cpufreq/scaling_cur_freq")
                ?: readLong("/sys/devices/system/cpu/cpu$core/cpufreq/cpuinfo_cur_freq")
            currentKhz?.let { core to it }
        }.toMap()
        val processCpu = Process.getElapsedCpuTime()
        synchronized(lock) {
            samples += CpuSample(counters, frequencies, processCpu, elapsed)
            val mem = readMemoryInfo()
            lastMemoryInfo = mem
            peakRssKb = listOfNotNull(peakRssKb, mem["VmRSS"]).maxOrNull()
            readThermalSensors().forEach { (name, celsius) ->
                thermalSamples.getOrPut(name) { mutableListOf() }.add(celsius)
            }
            lastPsi = readPsi()
        }
    }

    private fun readCpuCounters(): Map<Int, CpuCounters> = runCatching {
        File("/proc/stat").readLines().mapNotNull { line ->
            val parts = line.trim().split(Regex("\\s+"))
            val match = Regex("cpu(\\d+)").matchEntire(parts.firstOrNull() ?: "") ?: return@mapNotNull null
            val values = parts.drop(1).mapNotNull { it.toLongOrNull() }
            if (values.size < 4) return@mapNotNull null
            val idle = values[3] + (values.getOrNull(4) ?: 0L)
            match.groupValues[1].toInt() to CpuCounters(values.sum(), idle)
        }.toMap()
    }.getOrDefault(emptyMap())

    private fun readMemoryInfo(): Map<String, Long> {
        val result = mutableMapOf<String, Long>()
        runCatching {
            File("/proc/meminfo").readLines().forEach { line ->
                val parts = line.split(Regex("\\s+"))
                val key = parts.firstOrNull()?.removeSuffix(":") ?: return@forEach
                parts.getOrNull(1)?.toLongOrNull()?.let { result[key] = it }
            }
        }
        runCatching {
            File("/proc/self/status").readLines().forEach { line ->
                val parts = line.split(Regex("\\s+"))
                val key = parts.firstOrNull()?.removeSuffix(":") ?: return@forEach
                if (key in setOf("VmRSS", "VmHWM", "Threads")) {
                    parts.getOrNull(1)?.toLongOrNull()?.let { result[key] = it }
                }
            }
        }
        return result
    }

    private fun readThermalSensors(): Map<String, Double> {
        val result = mutableMapOf<String, Double>()
        runCatching {
            File("/sys/class/thermal").listFiles()
                ?.filter { it.name.startsWith("thermal_zone") }
                ?.forEach { zone ->
                    val raw = readLong(File(zone, "temp").path) ?: return@forEach
                    val type = readText(File(zone, "type").path)?.replace(Regex("[^A-Za-z0-9_.-]"), "_")
                        ?.takeIf { it.isNotBlank() } ?: zone.name
                    result[type] = normalizeTemperature(raw)
                }
        }
        readLong("/sys/class/power_supply/battery/temp")?.let { result["battery"] = if (it > 1000) it / 1000.0 else it / 10.0 }
        return result
    }

    private fun readPsi(): String? = readText("/proc/pressure/memory")
        ?.lineSequence()?.firstOrNull { it.startsWith("some ") }
        ?.let { line -> Regex("avg10=([0-9.]+).*avg60=([0-9.]+).*avg300=([0-9.]+)").find(line)?.value ?: line }

    private fun normalizeTemperature(value: Long): Double = when {
        value >= 1000 -> value / 1000.0
        value > 200 -> value / 10.0
        else -> value.toDouble()
    }

    private fun readLong(path: String): Long? = readText(path)?.trim()?.toLongOrNull()
    private fun readText(path: String): String? = runCatching { File(path).takeIf { it.canRead() }?.readText() }.getOrNull()

    companion object {
        private const val SAMPLE_INTERVAL_MS = 1000L
        fun start(): AndroidDevicePerformanceMonitor = AndroidDevicePerformanceMonitor()
    }
}
