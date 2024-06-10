use pgrx::pg_sys;
use std::os::raw::c_int;

#[repr(C)]
pub struct JitInstrumentation {
    pub created_functions: pg_sys::__ssize_t,
    pub generation_counter: pg_sys::instr_time,
    pub inlining_counter: pg_sys::instr_time,
    pub optimization_counter: pg_sys::instr_time,
    pub emission_counter: pg_sys::instr_time,
}

#[repr(C)]
pub struct JitContext {
    pub flags: c_int,
    pub resowner: *mut pg_sys::ResourceOwnerData,
    pub instr: JitInstrumentation,
}

#[repr(C)]
pub struct JitProviderCallbacks {
    pub reset_after_error: Option<unsafe extern "C" fn()>,
    pub release_context: Option<unsafe extern "C" fn(context: *mut pg_sys::JitContext)>,
    pub compile_expr: Option<unsafe extern "C" fn(state: *mut pg_sys::ExprState) -> bool>,
}

extern "C" {
    pub fn ResourceOwnerEnlargeJIT(owner: *mut pg_sys::ResourceOwnerData);
    pub fn ResourceOwnerRememberJIT(owner: *mut pg_sys::ResourceOwnerData, handle: pg_sys::Datum);

    pub static mut jit_dump_bitcode: bool;
}
