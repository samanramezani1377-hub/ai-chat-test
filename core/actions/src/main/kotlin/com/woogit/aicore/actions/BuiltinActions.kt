package com.woogit.aicore.actions

import com.woogit.aicore.domain.Action
import com.woogit.aicore.domain.ActionArgument
import com.woogit.aicore.domain.ActionArgumentType
import com.woogit.aicore.domain.ActionSchema
import com.woogit.aicore.domain.RiskLevel
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private const val MAX_FILE_BYTES = 1_048_576L

internal object ActionArguments {
    fun string(input: Any, name: String): String {
        val raw = input.toString()
        var index = 0
        fun skipWhitespace() { while (index < raw.length && raw[index].isWhitespace()) index++ }
        fun consume(char: Char): Boolean {
            if (index < raw.length && raw[index] == char) { index++; return true }
            return false
        }
        fun expect(char: Char) { require(consume(char)) { "Invalid JSON action arguments" } }
        fun quotedEnd(start: Int): Int {
            require(start < raw.length && raw[start] == '"') { "Expected JSON string" }
            var escaped = false
            for (i in start + 1 until raw.length) {
                val char = raw[i]
                if (escaped) escaped = false
                else if (char == '\\') escaped = true
                else if (char == '"') return i
                else require(char.code >= 0x20) { "Invalid control character in JSON string" }
            }
            throw IllegalArgumentException("Unterminated JSON string")
        }
        fun decodeQuoted(start: Int): Pair<String, Int> {
            val end = quotedEnd(start)
            return decode(raw.substring(start + 1, end), name) to (end + 1)
        }
        fun skipValue(): Int {
            require(index < raw.length) { "Missing JSON value" }
            when (raw[index]) {
                '"' -> return quotedEnd(index) + 1
                '{', '[' -> {
                    val stack = ArrayDeque<Char>()
                    stack.addLast(if (raw[index] == '{') '}' else ']')
                    var i = index + 1
                    var inString = false
                    var escaped = false
                    while (i < raw.length && stack.isNotEmpty()) {
                        val char = raw[i]
                        if (inString) {
                            if (escaped) escaped = false
                            else if (char == '\\') escaped = true
                            else if (char == '"') inString = false
                        } else {
                            when (char) {
                                '"' -> inString = true
                                '{' -> stack.addLast('}')
                                '[' -> stack.addLast(']')
                                '}', ']' -> {
                                    require(stack.removeLast() == char) { "Mismatched JSON delimiters" }
                                }
                            }
                        }
                        i++
                    }
                    require(stack.isEmpty() && !inString) { "Unterminated JSON value" }
                    return i
                }
                else -> {
                    val start = index
                    while (index < raw.length && raw[index] != ',' && raw[index] != '}' && !raw[index].isWhitespace()) index++
                    require(index > start) { "Missing JSON value" }
                    return index
                }
            }
        }

        skipWhitespace()
        expect('{')
        skipWhitespace()
        var found: String? = null
        var foundKey = false
        if (consume('}')) {
            require(name.isEmpty()) { "Missing string argument: " + name }
            return ""
        }
        while (true) {
            skipWhitespace()
            val (key, next) = decodeQuoted(index)
            index = next
            skipWhitespace()
            expect(':')
            skipWhitespace()
            if (key == name) {
                require(!foundKey) { "Duplicate JSON argument: " + name }
                require(index < raw.length && raw[index] == '"') { "Argument must be a JSON string: " + name }
                val (value, valueEnd) = decodeQuoted(index)
                found = value
                foundKey = true
                index = valueEnd
            } else {
                index = skipValue()
            }
            skipWhitespace()
            if (consume('}')) break
            expect(',')
        }
        skipWhitespace()
        require(index == raw.length) { "Trailing data in JSON action arguments" }
        return found ?: throw IllegalArgumentException("Missing string argument: " + name)
    }

    private fun decode(value: String, name: String): String {
        val out = StringBuilder(value.length)
        var i = 0
        while (i < value.length) {
            val c = value[i]
            if (c != '\\') { out.append(c); i++; continue }
            if (++i >= value.length) throw IllegalArgumentException("Invalid escape in argument: " + name)
            when (val escaped = value[i]) {
                '"', '\\', '/' -> out.append(escaped)
                'b' -> out.append('\b')
                'f' -> out.append('\u000C')
                'n' -> out.append('\n')
                'r' -> out.append('\r')
                't' -> out.append('\t')
                'u' -> {
                    require(i + 4 < value.length) { "Invalid unicode escape in argument: " + name }
                    val code = value.substring(i + 1, i + 5).toIntOrNull(16)
                        ?: throw IllegalArgumentException("Invalid unicode escape in argument: " + name)
                    out.append(code.toChar())
                    i += 4
                }
                else -> throw IllegalArgumentException("Invalid escape in argument: " + name)
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
    override val description = "Evaluate a basic arithmetic expression safely; no code execution."
    override val schema = ActionSchema(arguments = listOf(ActionArgument("expression", ActionArgumentType.STRING, maxLength = 512)))
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
    override val description = "Return the device current date and time with its UTC offset. Takes no arguments."
    override val schema = ActionSchema()
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
    override val description = "Create a UTF-8 text file inside the app Workspace. Paths are relative; existing files with different content are not overwritten."
    override val schema = ActionSchema(arguments = listOf(ActionArgument("file_name", ActionArgumentType.STRING, maxLength = 1024), ActionArgument("content", ActionArgumentType.STRING, maxLength = 1_048_576)))
    override val risk = RiskLevel.NORMAL
    override val stateChanging = true
    override val idempotent = true
    override suspend fun execute(input: Any): Any {
        val relativeName = ActionArguments.string(input, "file_name")
        var path = workspace.resolve(relativeName)
        val content = ActionArguments.string(input, "content")
        val bytes = content.toByteArray(StandardCharsets.UTF_8)
        require(bytes.size <= MAX_FILE_BYTES) { "File content exceeds 1 MiB" }
        Files.createDirectories(path.parent)
        // Re-check after parent creation so a newly introduced symlink is rejected.
        path = workspace.resolve(relativeName)
        require(!Files.isSymbolicLink(path)) { "Symbolic links cannot be modified" }

        if (Files.exists(path, LinkOption.NOFOLLOW_LINKS)) {
            require(Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) { "Existing target is not a regular file" }
            val existing = Files.readAllBytes(path)
            require(existing.contentEquals(bytes)) { "File already exists with different content; refusing overwrite" }
            return "already_exists:${workspace.relative(path)}"
        }

        val temp = Files.createTempFile(path.parent, ".woogit-", ".tmp")
        try {
            Files.write(temp, bytes)
            require(Files.size(temp) == bytes.size.toLong()) { "File write verification failed" }
            try {
                // Do not request ATOMIC_MOVE here: some providers may replace an existing
                // destination during an atomic move, which violates create_file semantics.
                Files.move(temp, path)
            } catch (_: java.nio.file.FileAlreadyExistsException) {
                require(Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) { "Existing target is not a regular file" }
                val existing = Files.readAllBytes(path)
                require(existing.contentEquals(bytes)) { "File already exists with different content; refusing overwrite" }
                return "already_exists:${workspace.relative(path)}"
            }
            require(Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) { "File creation verification failed" }
            require(Files.size(path) == bytes.size.toLong()) { "File write verification failed" }
        } finally {
            Files.deleteIfExists(temp)
        }
        return "created:${workspace.relative(path)}"
    }
}

class ReadFileAction(private val workspace: WorkspacePathResolver) : Action<Any, Any> {
    override val id = "read_file"
    override val description = "Read a UTF-8 text file inside the app Workspace. Paths are relative; files larger than 1 MiB are rejected."
    override val schema = ActionSchema(arguments = listOf(ActionArgument("file_name", ActionArgumentType.STRING, maxLength = 1024)))
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
    override val description = "List names of files and directories in the app Workspace. Optional path is relative; omit it to list the Workspace root."
    override val schema = ActionSchema(arguments = listOf(ActionArgument("path", ActionArgumentType.STRING, required = false, maxLength = 1024)))
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
    override val description = "Permanently delete a regular file inside the app Workspace. Sensitive action; requires explicit user approval."
    override val schema = ActionSchema(arguments = listOf(ActionArgument("file_name", ActionArgumentType.STRING, maxLength = 1024)))
    override val risk = RiskLevel.SENSITIVE
    override val stateChanging = true
    override val idempotent = false
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
