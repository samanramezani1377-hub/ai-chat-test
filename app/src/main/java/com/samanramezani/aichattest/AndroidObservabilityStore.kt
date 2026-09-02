package com.samanramezani.aichattest

import android.content.Context
import com.woogit.aicore.observability.ErrorReport
import com.woogit.aicore.observability.ErrorReportStore
import com.woogit.aicore.observability.ErrorReportFormatter
import com.woogit.aicore.observability.ExecutionTraceEvent
import com.woogit.aicore.observability.ExecutionTraceStore
import com.woogit.aicore.observability.Severity
import com.woogit.aicore.observability.RedactionStatus
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONArray
import org.json.JSONObject
import java.time.Instant

/** Small durable store for Error Center data. Retains only the newest 200 records. */
class AndroidErrorReportStore(context: Context) : ErrorReportStore {
    private val prefs = context.applicationContext.getSharedPreferences("central_observability", Context.MODE_PRIVATE)
    private val mutex = Mutex()
    private val key = "error_reports_v1"

    override suspend fun append(report: ErrorReport) = mutex.withLock {
        val values = read().toMutableList()
        values += report
        prefs.edit().putString(key, JSONArray(values.takeLast(200).map(::toJson)).toString()).apply()
    }

    override suspend fun recent(limit: Int?): List<ErrorReport> = mutex.withLock {
        val values = read().asReversed()
        if (limit == null) values else values.take(limit.coerceAtLeast(0))
    }

    override suspend fun find(reportId: String): ErrorReport? = mutex.withLock { read().firstOrNull { it.reportId == reportId } }

    private fun read(): List<ErrorReport> {
        val raw = prefs.getString(key, null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(raw)
            (0 until array.length()).mapNotNull { index -> fromJson(array.getJSONObject(index)) }
        }.getOrDefault(emptyList())
    }

    private fun toJson(r: ErrorReport) = JSONObject().apply {
        put("reportId", r.reportId); put("appVersion", r.appVersion); put("timestamp", r.timestamp.toString())
        put("severity", r.severity.name); put("component", r.component); put("errorCode", r.errorCode); put("eventId", r.eventId)
        putOpt("taskId", r.taskId); putOpt("actionId", r.actionId); put("userMessageFa", r.userMessageFa); put("rawMachineError", r.rawMachineError)
        putOpt("trace", r.trace); put("relatedLogs", JSONArray(r.relatedLogs)); putOpt("executionState", r.executionState)
        putOpt("recoveryState", r.recoveryState); putOpt("verificationState", r.verificationState); putOpt("runtimeInfo", r.runtimeInfo)
        putOpt("modelInfo", r.modelInfo); put("redactionStatus", r.redactionStatus.name)
    }

    private fun fromJson(o: JSONObject) = runCatching {
        ErrorReport(
            reportId = o.getString("reportId"), appVersion = o.getString("appVersion"), timestamp = Instant.parse(o.getString("timestamp")),
            severity = Severity.valueOf(o.getString("severity")), component = o.getString("component"), errorCode = o.getString("errorCode"),
            eventId = o.getString("eventId"), taskId = o.optString("taskId").takeIf { it.isNotEmpty() }, actionId = o.optString("actionId").takeIf { it.isNotEmpty() },
            userMessageFa = o.getString("userMessageFa"), rawMachineError = o.getString("rawMachineError"), trace = o.optString("trace").takeIf { it.isNotEmpty() },
            relatedLogs = (o.optJSONArray("relatedLogs") ?: JSONArray()).let { a -> (0 until a.length()).map { a.getString(it) } },
            executionState = o.optString("executionState").takeIf { it.isNotEmpty() }, recoveryState = o.optString("recoveryState").takeIf { it.isNotEmpty() },
            verificationState = o.optString("verificationState").takeIf { it.isNotEmpty() }, runtimeInfo = o.optString("runtimeInfo").takeIf { it.isNotEmpty() },
            modelInfo = o.optString("modelInfo").takeIf { it.isNotEmpty() }, redactionStatus = RedactionStatus.valueOf(o.getString("redactionStatus")),
        )
    }.getOrNull()
}

/** Durable execution trace store with the same retention boundary as Error Center. */
class AndroidExecutionTraceStore(context: Context) : ExecutionTraceStore {
    private val prefs = context.applicationContext.getSharedPreferences("central_observability", Context.MODE_PRIVATE)
    private val mutex = Mutex()
    private val key = "execution_trace_v1"

    override suspend fun append(event: ExecutionTraceEvent) = mutex.withLock {
        val values = read().toMutableList()
        values += event
        prefs.edit().putString(key, JSONArray(values.takeLast(1000).map(::toJson)).toString()).apply()
    }

    override suspend fun forExecution(executionId: String): List<ExecutionTraceEvent> = mutex.withLock { read().filter { it.executionId == executionId } }

    private fun read(): List<ExecutionTraceEvent> {
        val raw = prefs.getString(key, null) ?: return emptyList()
        return runCatching {
            val a = JSONArray(raw)
            (0 until a.length()).mapNotNull { fromJson(a.getJSONObject(it)) }
        }.getOrDefault(emptyList())
    }

    private fun toJson(e: ExecutionTraceEvent) = JSONObject().apply {
        put("eventId", e.eventId); put("executionId", e.executionId); putOpt("taskId", e.taskId); putOpt("actionId", e.actionId)
        put("phase", e.phase.name); putOpt("message", e.message); put("timestamp", e.timestamp.toString())
    }

    private fun fromJson(o: JSONObject) = runCatching {
        ExecutionTraceEvent(
            eventId = o.getString("eventId"), executionId = o.getString("executionId"), taskId = o.optString("taskId").takeIf { it.isNotEmpty() },
            actionId = o.optString("actionId").takeIf { it.isNotEmpty() }, phase = ExecutionTraceEvent.Phase.valueOf(o.getString("phase")),
            message = o.optString("message").takeIf { it.isNotEmpty() }, timestamp = Instant.parse(o.getString("timestamp")),
        )
    }.getOrNull()
}

fun ErrorReport.toCopyText(): String = ErrorReportFormatter.forAgent(this)
