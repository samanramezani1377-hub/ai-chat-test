package com.samanramezani.aichattest.screen

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * Replaces a file without deleting the last known-good destination first.
 * Both files must be in the same directory so rename/move stays on one filesystem.
 */
internal object AtomicFileReplacement {
    fun replace(source: File, destination: File) {
        require(source.isFile) { "Replacement source does not exist: ${source.name}" }
        val parent = destination.absoluteFile.parentFile
            ?: error("Replacement destination has no parent directory.")
        check(source.absoluteFile.parentFile == parent) {
            "Replacement source and destination must share a directory."
        }

        try {
            Files.move(
                source.toPath(),
                destination.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
            return
        } catch (_: Throwable) {
            // Some Android filesystems do not support ATOMIC_MOVE with replacement.
            // Fall through to a rollback-safe same-directory rename.
        }

        val backup = File.createTempFile(destination.name + ".", ".backup", parent)
        check(backup.delete()) { "Could not prepare replacement backup." }
        val hadPrevious = destination.exists()
        if (hadPrevious) {
            check(destination.renameTo(backup)) { "Could not preserve the previous model file." }
        }
        try {
            check(source.renameTo(destination)) { "Could not move the replacement model file." }
            if (hadPrevious) backup.delete()
        } catch (failure: Throwable) {
            if (hadPrevious) {
                if (destination.exists()) destination.delete()
                check(backup.renameTo(destination)) {
                    "Replacement failed and restoring the previous model file also failed."
                }
            }
            throw failure
        }
    }
}
