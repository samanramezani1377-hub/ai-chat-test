package com.woogit.aicore.actions

import com.woogit.aicore.domain.VerificationResult
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.time.Instant

/** Verifies observable postconditions of builtin actions instead of trusting their return strings. */
class BuiltinActionVerifier(private val workspaceRoot: Path) : ActionResultVerifier {
    private val workspace = WorkspacePathResolver(workspaceRoot)

    override suspend fun verify(action: PreparedAction, result: Any): VerificationResult = when (action.actionId) {
        "create_file" -> verifyCreate(action, result)
        "read_file" -> verifyRead(action, result)
        "list_files" -> verifyList(result)
        "delete_file" -> verifyDelete(action, result)
        "calculate" -> verifyCalculate(result)
        "get_time" -> verifyTime(result)
        "get_model_info", "get_performance_stats", "get_device_info" ->
            if (result.toString().isNotBlank()) VerificationResult(true, result.toString())
            else VerificationResult(false, "Provider returned an empty result")
        else -> VerificationResult(false, "No independent verifier registered for action: ${action.actionId}")
    }

    private fun verifyCreate(action: PreparedAction, result: Any): VerificationResult {
        val name = runCatching { ActionArguments.string(action.input, "file_name") }.getOrNull()
            ?: return VerificationResult(false, "Missing file_name for verification")
        val content = runCatching { ActionArguments.string(action.input, "content") }.getOrNull()
            ?: return VerificationResult(false, "Missing content for verification")
        val path = runCatching { workspace.resolve(name) }.getOrNull()
            ?: return VerificationResult(false, "Invalid workspace path")
        if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) return VerificationResult(false, "Created path is not a regular file")
        val expected = content.toByteArray(StandardCharsets.UTF_8)
        val actual = runCatching { Files.readAllBytes(path) }.getOrNull()
            ?: return VerificationResult(false, "Created file cannot be read back")
        if (!actual.contentEquals(expected)) return VerificationResult(false, "Created file content does not match requested content")
        return VerificationResult(true, "verified:create_file:${workspace.relative(path)}:${actual.size} bytes")
    }

    private fun verifyRead(action: PreparedAction, result: Any): VerificationResult {
        val name = runCatching { ActionArguments.string(action.input, "file_name") }.getOrNull()
            ?: return VerificationResult(false, "Missing file_name for verification")
        val path = runCatching { workspace.resolve(name) }.getOrNull()
            ?: return VerificationResult(false, "Invalid workspace path")
        if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) return VerificationResult(false, "Read target no longer exists")
        val actual = runCatching { Files.readString(path, StandardCharsets.UTF_8) }.getOrNull()
            ?: return VerificationResult(false, "Read target cannot be read back")
        return if (actual == result.toString()) VerificationResult(true, "verified:read_file:${workspace.relative(path)}")
        else VerificationResult(false, "Read result does not match current file contents")
    }

    private fun verifyList(result: Any): VerificationResult =
        VerificationResult(result.toString().lines().all { it.isBlank() || !it.contains("..${java.io.File.separator}") }, "verified:list_files")

    private fun verifyDelete(action: PreparedAction, result: Any): VerificationResult {
        val name = runCatching { ActionArguments.string(action.input, "file_name") }.getOrNull()
            ?: return VerificationResult(false, "Missing file_name for verification")
        val path = runCatching { workspace.resolve(name) }.getOrNull()
            ?: return VerificationResult(false, "Invalid workspace path")
        return if (!Files.exists(path, LinkOption.NOFOLLOW_LINKS)) VerificationResult(true, "verified:delete_file:${workspace.relative(path)}")
        else VerificationResult(false, "Deleted path still exists")
    }

    private fun verifyCalculate(result: Any): VerificationResult =
        if (result.toString().toDoubleOrNull()?.isFinite() == true) VerificationResult(true, "verified:calculate")
        else VerificationResult(false, "Calculation result is not a finite number")

    private fun verifyTime(result: Any): VerificationResult =
        runCatching { Instant.parse(result.toString()) }
            .fold(onSuccess = { VerificationResult(true, "verified:get_time") }, onFailure = { VerificationResult(false, "Invalid time result") })
}
