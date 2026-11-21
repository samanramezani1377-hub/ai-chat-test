# Shared strict source-patching helper for pinned upstream dependencies.
function(ai_chat_replace_once var_name old_text new_text description)
    string(FIND "${${var_name}}" "${old_text}" _patch_pos)
    if (_patch_pos EQUAL -1)
        message(FATAL_ERROR "Could not apply no-ShaderF16 WebGPU patch: ${description}")
    endif()
    string(REPLACE "${old_text}" "${new_text}" _patch_result "${${var_name}}")
    set(${var_name} "${_patch_result}" PARENT_SCOPE)
endfunction()
