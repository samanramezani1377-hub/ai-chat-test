package com.samanramezani.aichattest.storage

import java.io.File
import java.util.UUID

/**
 * Replaces a validated directory while retaining the previous copy until the new
 * directory has been moved into place. Source must be inside the destination parent
 * tree so both renames stay on the same filesystem.
 */
internal object SafeDirectoryReplacement {
    fun replace(source: File, destination: File) {
        require(source.isDirectory) { "Replacement source directory does not exist: ${source.name}" }
        val parent = destination.absoluteFile.parentFile
            ?: error("Replacement destination has no parent directory.")
        val parentPath = parent.canonicalFile.toPath()
        check(source.canonicalFile.toPath().startsWith(parentPath)) {
            "Replacement source must be staged under the destination parent."
        }
        check(source.absoluteFile != destination.absoluteFile) {
            "Replacement source and destination must be different paths."
        }

        val backup = File(parent, ".${destination.name}.${UUID.randomUUID()}.backup")
        check(!backup.exists()) { "Could not reserve a directory replacement backup path." }
        val hadPrevious = destination.exists()
        if (hadPrevious) {
            check(destination.renameTo(backup)) {
                "Could not preserve the previous directory."
            }
        }

        try {
            check(source.renameTo(destination)) {
                "Could not move the replacement directory into place."
            }
            if (hadPrevious) backup.deleteRecursively()
        } catch (failure: Throwable) {
            if (hadPrevious) {
                if (destination.exists()) {
                    check(destination.deleteRecursively()) {
                        "Could not remove the incomplete replacement directory."
                    }
                }
                check(backup.renameTo(destination)) {
                    "Replacement failed and restoring the previous directory also failed."
                }
            }
            throw failure
        }
    }
}
