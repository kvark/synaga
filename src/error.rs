#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Syn(#[from] syn::Error),
    /// Where in the source the error below came from.
    ///
    /// Attached per item, which is as fine-grained as the lowering gets: it
    /// names the `fn` or `struct` and the line it opens on, so a failure in a
    /// build step points somewhere rather than just failing.
    #[error("{item} on line {line}: {source}")]
    At {
        item: String,
        line: usize,
        #[source]
        source: Box<Error>,
    },
    #[error("unsupported item: {0}")]
    UnsupportedItem(String),
    #[error("unsupported type: {0}")]
    UnsupportedType(String),
    #[error("unsupported method `{0}`")]
    UnsupportedMethod(String),
    #[error("unsupported expression: {0}")]
    UnsupportedExpr(String),
    #[error("unsupported binary operator: {0}")]
    UnsupportedBinOp(String),
    #[error("unsupported statement: {0}")]
    UnsupportedStmt(String),
    #[error("unknown identifier: {0}")]
    UnknownIdent(String),
    #[error("function `{0}` has no return type")]
    MissingReturnType(String),
    #[error("function `{0}` can finish without returning a value")]
    MissingReturn(String),
    #[error("receiver arguments are not supported")]
    Receiver,
    #[error("pattern parameters are not supported")]
    PatternParam,
    #[error("`let` without initializer is not supported")]
    MissingLetInit,
    #[error("`if` used as a value needs an `else` branch")]
    IfExprMissingElse,
    #[error("block used as a value has no tail expression")]
    MissingBlockValue,
    #[error("type mismatch")]
    TypeMismatch,
    #[error("operator `{0}` does not apply to these operand types")]
    BadOperandTypes(String),
    #[error("shift amount must be `u32` and match the left operand's size")]
    BadShiftType,
    #[error("unsupported cast to `{0}`")]
    UnsupportedCast(String),
    #[error("cannot assign to function argument `{0}`")]
    AssignToArgument(String),
    #[error("cannot assign to this expression (a swizzle or call result is not storage)")]
    InvalidAssignTarget,
    #[error("labeled loops are not supported")]
    LoopLabel,
    #[error("`break` with a value is not supported")]
    BreakValue,
    #[error("unsupported vector constructor `{0}`")]
    BadVecCtor(String),
    #[error("wrong number of components for vector constructor")]
    VecCtorArgs,
    #[error("unsupported swizzle `.{0}`")]
    UnsupportedSwizzle(String),
    #[error("vector component index out of range")]
    VecIndexRange,
    #[error("`#[entry_point]` needs a stage first: `vertex`, `fragment` or `compute`")]
    UnknownStage,
    #[error("`#[{0}]` is spelled `#[entry_point(..)]` now: `#[entry_point(vertex)]`, `#[entry_point(compute, threads(8, 8))]`")]
    OldStageAttribute(String),
    #[error("duplicate `#[{0}]` attribute")]
    DuplicateAttribute(String),
    #[error("unsupported binding `{0}`")]
    UnsupportedBinding(String),
    #[error("entry point argument `{0}` needs #[location] or #[builtin]")]
    MissingArgBinding(String),
    #[error("a compute entry point needs `threads(x, y, z)`")]
    MissingWorkgroupSize,
    #[error("`threads` is only for a compute entry point")]
    UnexpectedWorkgroupSize,
    #[error("entry point `{0}` is missing #[output(...)]")]
    MissingReturnBinding(String),
    #[error("`{0}` produces no value, so it cannot be used as one")]
    ValueFromStatement(String),
    #[error("unexpected address space on `{0}`: a texture or sampler is a handle")]
    UnexpectedAddressSpace(String),
    #[error("`{0}` needs an `acceleration_structure`")]
    NotAnAccelerationStructure(String),
    #[error("`{0}` is workgroup or private memory, so it takes no binding")]
    UnexpectedBinding(String),
    #[error("`{0}` is a texture method, and this is not a texture")]
    NotATexture(String),
    #[error("unknown function `{0}`")]
    UnknownFunction(String),
    #[error("duplicate function `{0}`")]
    DuplicateFunction(String),
    #[error("wrong number of arguments for `{0}`")]
    WrongArgCount(String),
    #[error("unsupported matrix constructor `{0}`")]
    BadMatCtor(String),
    #[error("wrong number of components for matrix constructor")]
    MatCtorArgs,
    #[error("resource `{0}` needs #[group] and #[binding]")]
    MissingResourceBinding(String),
    #[error("`{0}` is a runtime-sized array, so it needs `#[storage]`")]
    RuntimeArrayNotStorage(String),
    #[error("duplicate global `{0}`")]
    DuplicateGlobal(String),
    #[error("cannot write through `{0}`, which is read-only")]
    AssignToReadonly(String),
    #[error("duplicate const `{0}`")]
    DuplicateConst(String),
    #[error("`{0}` is not allowed in a constant")]
    UnsupportedConstExpr(String),
    #[error("duplicate struct `{0}`")]
    DuplicateStruct(String),
    #[error("unknown struct `{0}`")]
    UnknownStruct(String),
    #[error("struct `{0}` has no fields")]
    EmptyStruct(String),
    #[error("duplicate field `{0}`")]
    DuplicateField(String),
    #[error("unknown field `{0}`")]
    UnknownField(String),
    #[error("missing field `{0}`")]
    MissingStructField(String),
    #[error("wrong number of fields for struct `{0}`")]
    StructFieldCount(String),
    #[error("struct `{0}` mixes bound and unbound fields")]
    MixedStructBindings(String),
    #[error("entry point `{0}` returns a struct with field bindings; drop `#[output(...)]`")]
    RedundantReturnBinding(String),
    #[error("`#[location]` field `{0}` is an integer, so it needs `#[flat]`")]
    MissingFlat(String),
    #[error("unsupported `cfg` predicate `{0}`")]
    UnsupportedCfg(String),
    #[error("`{0}` could mean items in more than one module; import the one you mean")]
    AmbiguousName(String),
    #[error("`{0}` depends on itself; a shader cannot recurse, and a type cannot contain itself")]
    Cycle(String),
    #[error("`{0}` is an entry point, which only the GPU calls")]
    CallToEntryPoint(String),
}

impl Error {
    /// Where in the source this happened, as a 1-based line and column.
    ///
    /// A parse error knows both; a lowering error knows the line of the item
    /// it came from. Anything the lowering raises outside an item knows
    /// neither.
    pub fn location(&self) -> Option<(usize, usize)> {
        match self {
            Error::Syn(err) => {
                let start = err.span().start();
                Some((start.line, start.column + 1))
            }
            Error::At { line, .. } => Some((*line, 1)),
            _ => None,
        }
    }
}
