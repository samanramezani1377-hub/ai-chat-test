use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JString};
use jni::sys::{jboolean, jint, jintArray, jlong};
use tokenizers::Tokenizer;

fn throw(env: &mut JNIEnv, message: &str) {
    let _ = env.throw_new("java/lang/IllegalArgumentException", message);
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_samanramezani_aichattest_voice_LayaNativeTokenizer_createTokenizer(
    mut env: JNIEnv,
    _: JClass,
    tokenizer_bytes: JByteArray,
) -> jlong {
    let bytes = match env.convert_byte_array(&tokenizer_bytes) {
        Ok(bytes) => bytes,
        Err(error) => {
            throw(&mut env, &format!("LAYA-TOKENIZER-NATIVE-001: cannot read tokenizer JSON: {error}"));
            return 0;
        }
    };
    match Tokenizer::from_bytes(&bytes) {
        Ok(tokenizer) => Box::into_raw(Box::new(tokenizer)) as jlong,
        Err(error) => {
            throw(&mut env, &format!("LAYA-TOKENIZER-NATIVE-002: invalid tokenizer JSON: {error}"));
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_samanramezani_aichattest_voice_LayaNativeTokenizer_encode(
    mut env: JNIEnv,
    _: JClass,
    tokenizer_ptr: jlong,
    text: JString,
    add_special_tokens: jboolean,
) -> jintArray {
    if tokenizer_ptr == 0 {
        throw(&mut env, "LAYA-TOKENIZER-NATIVE-003: tokenizer is closed.");
        return std::ptr::null_mut();
    }
    let text: String = match env.get_string(&text) {
        Ok(value) => value.into(),
        Err(error) => {
            throw(&mut env, &format!("LAYA-TOKENIZER-NATIVE-004: cannot read input text: {error}"));
            return std::ptr::null_mut();
        }
    };
    let tokenizer = unsafe { &*(tokenizer_ptr as *const Tokenizer) };
    let encoding = match tokenizer.encode(text, add_special_tokens != 0) {
        Ok(encoding) => encoding,
        Err(error) => {
            throw(&mut env, &format!("LAYA-TOKENIZER-NATIVE-005: tokenization failed: {error}"));
            return std::ptr::null_mut();
        }
    };
    let ids: Vec<jint> = encoding.get_ids().iter().map(|&id| id as jint).collect();
    let output = match env.new_int_array(ids.len() as jint) {
        Ok(array) => array,
        Err(error) => {
            throw(&mut env, &format!("LAYA-TOKENIZER-NATIVE-006: cannot allocate token IDs: {error}"));
            return std::ptr::null_mut();
        }
    };
    if let Err(error) = env.set_int_array_region(&output, 0, &ids) {
        throw(&mut env, &format!("LAYA-TOKENIZER-NATIVE-007: cannot copy token IDs: {error}"));
        return std::ptr::null_mut();
    }
    output.into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_samanramezani_aichattest_voice_LayaNativeTokenizer_tokenId(
    mut env: JNIEnv,
    _: JClass,
    tokenizer_ptr: jlong,
    token: JString,
) -> jint {
    if tokenizer_ptr == 0 {
        throw(&mut env, "LAYA-TOKENIZER-NATIVE-008: tokenizer is closed.");
        return -1;
    }
    let token: String = match env.get_string(&token) {
        Ok(value) => value.into(),
        Err(error) => {
            throw(&mut env, &format!("LAYA-TOKENIZER-NATIVE-009: cannot read special token: {error}"));
            return -1;
        }
    };
    let tokenizer = unsafe { &*(tokenizer_ptr as *const Tokenizer) };
    tokenizer.token_to_id(&token).map(|id| id as jint).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_samanramezani_aichattest_voice_LayaNativeTokenizer_deleteTokenizer(
    _: JNIEnv,
    _: JClass,
    tokenizer_ptr: jlong,
) {
    if tokenizer_ptr != 0 {
        drop(unsafe { Box::from_raw(tokenizer_ptr as *mut Tokenizer) });
    }
}
