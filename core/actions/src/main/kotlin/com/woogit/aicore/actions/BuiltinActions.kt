package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.RiskLevel
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private const val MAX_FILE_BYTES = 1_048_576L

/** Strictly extracts a named JSON string and decodes standard JSON escapes. */
internal object ActionArguments {
    fun string(input: Any, name: String): String {
        val raw = input.toString()
        val marker = Regex("\\\"${Regex.escape(name)}\\\"\\s*:").find(raw)
            ?: throw IllegalArgumentException("Missing string argument: $name")
        var start = marker.range.last + 1
        while (start < raw.length && raw[start].isWhitespace()) start++
        require(start < raw.length && raw[start] == '\"') { "Argument must be a JSON string: $name" }
        var index = start + 1
        var escaped = false
        while (index < raw.length) {
            val c = raw[index]
            if (!escaped && c == '"') return decode(raw.substring(start + 1, index), name)
            if (escaped) escaped = false else if (c == '\\') escaped = true
            index++
        }
        throw IllegalArgumentException("Unterminated string argument: $name")
    }

    private fun decode(value: String, name: String): String {
        val out = StringBuilder(value.length)
        var i = 0
        while (i < value.length) {
            val c = value[i]
            if (c != '\\') { out.append(c); i++; continue }
            if (++i >= value.length) throw IllegalArgumentException("Invalid escape in argument: $name")
            when (val escaped = value[i]) {
                '"', '\\', '/' -> out.append(escaped)
                'b' -> out.append('\b')
                'f' -> out.append('\u000C')
                'n' -> out.append('\n')
                'r' -> out.append('\r')
                't' -> out.append('\t')
                'u' -> {
                    require(i + 4 < value.length) { "Invalid unicode escape in argument: $name" }
                    val code = value.substring(i + 1, i + 5).toIntOrNull(16)
                        ?: throw IllegalArgumentException("Invalid unicode escape in argument: $name")
                    out.append(code.toChar())
                    i += 4
                }
                else -> throw IllegalArgumentException("Invalid escape in argument: $name")
            }
            i++
        }
        return out.toString()
    }
}

private class ExpressionParser(private val source: String) {
    private var index = 0
    fun parse(): Double {
        skipWhitespace(); val value = expression(); skipWhitespace()
        require(index == source.length) { "Unexpected token at position $index" }
        require(value.isFinite()) { "Result is not finite" }
        return value
    }
    private fun expression(): Double {
        var value = term()
        while (true) { skipWhitespace(); value = when { consume('+') -> value + term(); consume('-') -> value - term(); else -> return value } }
    }
    private fun term(): Double {
        var value = unary()
        while (true) { skipWhitespace(); value = when { consume('*') -> value * unary(); consume('/') -> { val divisor = unary(); require(divisor != 0.0) { "Division by zero" }; value / divisor }; else -> return value } }
    }
    private fun unary(): Double { skipWhitespace(); return when { consume('+') -> unary(); consume('-') -> -unary(); else -> primary() } }
    private fun primary(): Double {
        skipWhitespace()
        if (consume('(')) { val value = expression(); skipWhitespace(); require(consume(')')) { "Missing ')'" }; return value }
        val start = index
        while (index < source.length && (source[index].isDigit() || source[index] == '.')) index++
        require(index > start) { "Expected number at position $index" }
        return source.substring(start, index).toDoubleOrNull() ?: throw IllegalArgumentException("Invalid number")
    }
    private fun consume(char: Char): Boolean { if (index < source.length && source[index] == char) { index++; return true }; return false }
    private fun skipWhitespace() { while (index < source.length && source[index].isWhitespace()) index++ }
}

class CalculateAction : Action<Any, Any> {
    override val id = "calculate"
    override val risk = RiskLevel.LOW
    override suspend fun execute(input: Any): Any {
        val expression = ActionArguments.string(input, "expression")
        require(expression.length <= 512) { "Expression is too long" }
        val result = ExpressionParser(expression).parse()
        return if (result % 1.0 == 0.0) result.toLong().toString() else result.toString()
    }
}

class GetTimeAction(private val clock: () -> Instant = { Instant.now() }, private val zoneId: ZoneId = ZoneId.systemDefault()) : Action<Any, Any> {
    override val id = "get_time"
    override val risk = RiskLevel.LOW
    override suspend fun execute(input: Any): Any {
        require(input.toString().trim().let { it == "{}" || it.isBlank() }) { "get_time accepts no arguments" }
        return DateTimeFormatter.ISO_OFFSET_DATE_TIME.format(clock().atZone(zoneId))
    }
}

class WorkspacePathResolver(private val root: Path) {
    private val rootPath: Path
    init {
        Files.createDirectories(root)
        require(Files.isDirectory(root, LinkOption.NOFOLLOW_LINKS)) { "Workspace root is not a directory" }
        require(!Files.isSymbolicLink(root)) { "Workspace root must not be a symbolic link" }
        rootPath = root.toAbsolutePath().normalize().toRealPath(LinkOption.NOFOLLOW_LINKS)
    }
    fun resolve(relativePath: String): Path {
        require(relativePath.isNotBlank()) { "Path must not be blank" }
        val candidate = rootPath.resolve(relativePath).normalize()
        require(candidate.startsWith(rootPath)) { "Path escapes workspace" }
        validateNoSymlinkTraversal(candidate)
        return candidate
    }
    fun relative(path: Path): String = rootPath.relativize(path.toAbsolutePath().normalize()).toString()
    private fun validateNoSymlinkTraversal(candidate: Path) {
        var current = rootPath
        for (part in rootPath.relativize(candidate)) {
            current = current.resolve(part)
            if (Files.exists(current, LinkOption.NOFOLLOW_LINKS) && Files.isSymbolicLink(current)) throw IllegalArgumentException("Symbolic links are not allowed inside workspace paths")
        }
    }
}

class CreateFileAction(private val workspace: WorkspacePathResolver) : Action<Any, Any> {
    override val id = "create_file"
    override val risk = RiskLevel.NORMAL
    override suspend fun execute(input: Any): Any {
        val path = workspace.resolve(ActionArguments.string(input, "file_name"))
        val content = ActionArguments.string(input, "content")
        val bytes = content.toByteArray(StandardCharsets.UTF_8)
        require(bytes.size <= MAX_FILE_BYTES) { "File content exceeds 1 MiB" }
        Files.createDirectories(path.parent)
        require(!Files.isSymbolicLink(path)) { "Symbolic links cannot be modified" }
        Files.write(path, bytes)
        require(Files.size(path) == bytes.size.toLong()) { "File write verification failed" }
        return "created:${workspace.relative(path)}"
    }
}

class ReadFileAction(private val workspace: WorkspacePathResolver) : Action<Any, Any> {
    override val id = "read_file"
    override val risk = RiskLevel.LOW
    override suspend fun execute(input: Any): Any {
        val path = workspace.resolve(ActionArguments.string(input, "file_name"))
        require(Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) { "File not found" }
        require(Files.size(path) <= MAX_FILE_BYTES) { "File exceeds 1 MiB" }
        return Files.readString(path, StandardCharsets.UTF_8)
    }
}

class ListFilesAction(private val workspace: WorkspacePathResolver) : Action<Any, Any> {
    override val id = "list_files"
    override val risk = RiskLevel.LOW
    override suspend fun execute(input: Any): Any {
        val relative = runCatching { ActionArguments.string(input, "path") }.getOrNull().orEmpty()
        val directory = workspace.resolve(relative.ifBlank { "." })
        require(Files.isDirectory(directory, LinkOption.NOFOLLOW_LINKS)) { "Directory not found" }
        Files.list(directory).use { stream -> return stream.filter { !Files.isSymbolicLink(it) }.map(workspace::relative).sorted().toList().joinToString("\n") }
    }
}

class DeleteFileAction(private val workspace: WorkspacePathResolver) : Action<Any, Any> {
    override val id = "delete_file"
    override val risk = RiskLevel.SENSITIVE
    override suspend fun execute(input: Any): Any {
        val path = workspace.resolve(ActionArguments.string(input, "file_name"))
        require(Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) { "File not found" }
        Files.delete(path)
        require(!Files.exists(path, LinkOption.NOFOLLOW_LINKS)) { "File deletion verification failed" }
        return "deleted:${workspace.relative(path)}"
    }
}

fun DefaultActionRegistry.registerBuiltinFileActions(workspaceRoot: Path) {
    val workspace = WorkspacePathResolver(workspaceRoot)
    register("utility", CalculateAction())
    register("utility", GetTimeAction())
    register("filesystem", CreateFileAction(workspace))
    register("filesystem", ReadFileAction(workspace))
    register("filesystem", ListFilesAction(workspace))
    register("filesystem", DeleteFileAction(workspace))
}
