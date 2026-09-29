//! `for` over a range, as Rust iterates one.
//!
//! An inclusive range runs for its end too, even when the end is the
//! largest value its type has, so it cannot stop by stepping past the end.
//! The binding is a copy of the range's position: assigning to it in
//! `for mut i in ..` changes `i`, not the iteration.

use synaga::naga::{self, BinaryOperator, Expression, Statement};

fn lower(src: &str) -> naga::Module {
    let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    synaga::validate(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    module
}

fn function<'a>(module: &'a naga::Module, name: &str) -> &'a naga::Function {
    module
        .functions
        .iter()
        .map(|(_, f)| f)
        .find(|f| f.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no function `{name}`"))
}

/// The loops in `block`, and whether each is under an `if`.
fn loops(block: &naga::Block) -> Vec<(bool, &naga::Block, &naga::Block, bool)> {
    fn walk<'a>(
        block: &'a naga::Block,
        guarded: bool,
        out: &mut Vec<(bool, &'a naga::Block, &'a naga::Block, bool)>,
    ) {
        for statement in block.iter() {
            match statement {
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    out.push((guarded, body, continuing, break_if.is_some()));
                    walk(body, guarded, out);
                }
                Statement::If { accept, reject, .. } => {
                    walk(accept, true, out);
                    walk(reject, true, out);
                }
                Statement::Block(inner) => walk(inner, guarded, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(block, false, &mut out);
    out
}

/// The comparisons `func` evaluates inside `block`, not counting nested blocks.
fn comparisons(func: &naga::Function, block: &naga::Block) -> Vec<BinaryOperator> {
    block
        .iter()
        .filter_map(|statement| match statement {
            Statement::Emit(range) => Some(range.clone()),
            _ => None,
        })
        .flatten()
        .filter_map(|h| match func.expressions[h] {
            Expression::Binary { op, .. }
                if matches!(
                    op,
                    BinaryOperator::Less
                        | BinaryOperator::LessEqual
                        | BinaryOperator::Equal
                        | BinaryOperator::NotEqual
                ) =>
            {
                Some(op)
            }
            _ => None,
        })
        .collect()
}

#[test]
fn an_inclusive_range_stops_after_its_end_without_stepping_past_it() {
    for (ty, range) in [
        ("u32", "u32::MAX..=u32::MAX"),
        ("i32", "(i32::MAX - 1)..=i32::MAX"),
        ("u32", "0..=n"),
    ] {
        let src = format!(
            "fn f(n: {ty}) -> {ty} {{ let mut last = n; for i in {range} {{ last = i; }} last }}"
        );
        let module = lower(&src);
        let f = function(&module, "f");
        let found = loops(&f.body);
        let [(guarded, body, continuing, break_if)] = found[..] else {
            panic!("{src}: {:#?}", f.body);
        };
        // An empty range, whose end is before its start, runs nothing.
        assert!(guarded, "{src}: {:#?}", f.body);
        // The loop leaves after the iteration for `end`, from `continuing`,
        // never by testing `i <= end` at the top.
        assert!(break_if, "{src}: {:#?}", f.body);
        assert!(comparisons(f, body).is_empty(), "{src}: {:#?}", body);
        assert_eq!(comparisons(f, continuing), [BinaryOperator::Equal], "{src}");
    }
}

#[test]
fn continue_on_the_last_iteration_still_leaves() {
    // `continue` goes to `continuing`, which is where the loop decides to
    // stop, so skipping the rest of the body cannot skip the stop.
    let module = lower(
        r#"
        fn f(n: u32) -> u32 {
            let mut odd = 0u32;
            for i in 0..=n {
                if i % 2 == 0 {
                    continue;
                }
                odd += 1;
            }
            odd
        }
        "#,
    );
    let f = function(&module, "f");
    let [(_, _, continuing, true)] = loops(&f.body)[..] else {
        panic!("{:#?}", f.body);
    };
    assert_eq!(comparisons(f, continuing), [BinaryOperator::Equal]);
}

#[test]
fn an_exclusive_range_tests_its_end_first() {
    let module = lower("fn f(n: u32) -> u32 { let mut t = 0u32; for i in 0..n { t += i; } t }");
    let f = function(&module, "f");
    let [(false, body, continuing, false)] = loops(&f.body)[..] else {
        panic!("{:#?}", f.body);
    };
    assert_eq!(comparisons(f, body), [BinaryOperator::Less]);
    assert!(comparisons(f, continuing).is_empty());
}

#[test]
fn a_mutable_binding_is_not_the_iteration() {
    let module = lower(
        r#"
        fn f(n: u32) -> u32 {
            let mut total = 0u32;
            for mut i in 0..n {
                i *= 10;
                total += i;
            }
            total
        }
        "#,
    );
    let f = function(&module, "f");
    // The counter, the copy the body changes, and `total`.
    let names: Vec<_> = f
        .local_variables
        .iter()
        .map(|(_, v)| v.name.as_deref().unwrap_or(""))
        .collect();
    assert_eq!(names.iter().filter(|&&n| n == "i").count(), 2, "{names:?}");
    // The counter is only stored to before the loop and in `continuing`.
    let [(_, body, _, _)] = loops(&f.body)[..] else {
        panic!("{:#?}", f.body);
    };
    let counter = f
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("i"))
        .map(|(h, _)| h)
        .unwrap();
    let stores_to_counter = body.iter().any(|statement| match statement {
        Statement::Store { pointer, .. } => {
            matches!(f.expressions[*pointer], Expression::LocalVariable(v) if v == counter)
        }
        _ => false,
    });
    assert!(!stores_to_counter, "{body:#?}");
}

#[test]
fn a_range_evaluates_its_ends_once() {
    let module = lower(
        r#"
        static counts: StorageMut<[u32]> = group(0).binding(0);
        fn next() -> u32 {
            counts.get_mut()[0] += 1;
            counts[0]
        }
        fn f() -> u32 {
            let mut t = 0u32;
            for i in next()..=next() {
                t += i;
            }
            t
        }
        "#,
    );
    let f = function(&module, "f");
    // Both calls come before the loop, one each.
    let calls = f
        .body
        .iter()
        .filter(|s| matches!(s, Statement::Call { .. }))
        .count();
    assert_eq!(calls, 2, "{:#?}", f.body);
    for (_, body, continuing, _) in loops(&f.body) {
        for block in [body, continuing] {
            assert!(
                !block.iter().any(|s| matches!(s, Statement::Call { .. })),
                "{block:#?}"
            );
        }
    }
}
