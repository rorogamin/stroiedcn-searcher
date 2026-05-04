#![allow(dead_code)]

use clap::{Parser, Subcommand};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::collections::hash_map::RandomState;
use std::fs;
use std::hash::{BuildHasher, Hasher};
use std::io::{self, Write};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

const ALPHABET: [u8; 9] = *b"STROIEDCN";
const DEFAULT_MAX_LEN: usize = 1_000_000;
const HASH1_MUL: u64 = 0x94D0_49BB_1331_11EB;
const HASH2_MUL: u64 = 0x369D_EA0F_31A5_3F85;

static HASH_SEEDS: OnceLock<(u64, u64)> = OnceLock::new();

#[inline(always)]
fn hash_seeds() -> (u64, u64) {
    *HASH_SEEDS.get_or_init(|| {
        let rs = RandomState::new();
        let mut h1 = rs.build_hasher();
        h1.write_u64(0x9E37_79B1_85EB_CA87);
        let seed1 = h1.finish();
        let mut h2 = rs.build_hasher();
        h2.write_u64(0xC2B2_AE3D_27D4_EB4F);
        let seed2 = h2.finish();
        (seed1, seed2)
    })
}
const INLINE_SEEN_CAP: usize = 128;
const TRUNCATED_RENDER_LEN: usize = 100;
const SEARCH_PREVIEW_LEN: usize = 40;
const STABLE_CYCLE_CHECK_STEPS: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Item {
    Comb(u8),
    Group(Arc<Expr>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Expr {
    items: Vec<Item>,
    flat_len: usize,
    fp1: u64,
    fp2: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Fingerprint {
    hi: u64,
    lo: u64,
}

const EMPTY_FINGERPRINT: Fingerprint = Fingerprint { hi: 0xDEAD_BEEF_CAFE_BABE, lo: 0xFEED_FACE_0BAD_F00D };

struct SeenFingerprints {
    inline: [Fingerprint; INLINE_SEEN_CAP],
    len: usize,
    spill: Option<FxHashSet<Fingerprint>>,
}

#[derive(Debug)]
pub struct ReduceResult {
    pub result: String,
    pub steps: usize,
    pub status: &'static str,
}

#[derive(Clone)]
struct SearchHit {
    expr: String,
    out: String,
    step: usize,
    status: &'static str,
}

struct CandidateResult {
    final_expr: Expr,
    step: usize,
    status: &'static str,
}

#[derive(Clone)]
struct SMatch {
    expr: String,
    steps: usize,
    status: &'static str,
}

struct SearchAccumulator {
    total_hits: usize,
    limit: usize,
    items: Vec<SearchHit>,
}

impl Item {
    #[inline(always)]
    fn flat_len(&self) -> usize {
        match self {
            Self::Comb(_) => 1,
            Self::Group(expr) => expr.flat_len,
        }
    }

    #[inline(always)]
    fn fp1(&self) -> u64 {
        match self {
            Self::Comb(c) => 0xA24B_AED4_963E_E407u64 ^ ((*c as u64).wrapping_mul(HASH1_MUL)),
            Self::Group(expr) => expr.fp1 ^ 0xC3A5_C85C_97CB_3127u64,
        }
    }

    #[inline(always)]
    fn fp2(&self) -> u64 {
        match self {
            Self::Comb(c) => 0x9FB2_1C65_1E98_DF25u64 ^ ((*c as u64).wrapping_mul(HASH2_MUL)),
            Self::Group(expr) => expr.fp2 ^ 0xB492_B66F_BE98_F273u64,
        }
    }
}

impl Expr {
    #[inline]
    fn new(items: Vec<Item>) -> Self {
        let mut expr = Self { items, flat_len: 0, fp1: 0, fp2: 0 };
        expr.refresh();
        expr
    }

    #[inline]
    fn from_index(mut index: usize, len: usize) -> Self {
        let mut items = vec![Item::Comb(ALPHABET[0]); len];
        for slot in items.iter_mut().rev() {
            *slot = Item::Comb(ALPHABET[index % ALPHABET.len()]);
            index /= ALPHABET.len();
        }
        Self::new(items)
    }

    #[inline(always)]
    fn fingerprint(&self) -> Fingerprint {
        Fingerprint {
            hi: self.fp1 ^ (self.flat_len as u64).wrapping_mul(HASH1_MUL),
            lo: self.fp2 ^ ((self.flat_len as u64).rotate_left(17)).wrapping_mul(HASH2_MUL),
        }
    }

    #[inline]
    fn refresh(&mut self) {
        let mut flat_len = 0usize;
        let (seed1, seed2) = hash_seeds();
        let mut fp1 = seed1 ^ (self.items.len() as u64);
        let mut fp2 = seed2 ^ ((self.items.len() as u64).rotate_left(17));

        for item in &self.items {
            flat_len += item.flat_len();
            fp1 = mix_hash(fp1, item.fp1(), HASH1_MUL);
            fp2 = mix_hash(fp2, item.fp2(), HASH2_MUL);
        }

        self.flat_len = flat_len;
        self.fp1 = fp1 ^ (flat_len as u64).wrapping_mul(HASH1_MUL);
        self.fp2 = fp2 ^ (flat_len as u64).wrapping_mul(HASH2_MUL);
    }

    #[inline]
    fn replace_consumed<I>(&mut self, pos: usize, consumed: usize, insert: I)
    where
        I: IntoIterator<Item = Item>,
    {
        self.items.splice(pos..pos + consumed, insert);
        self.refresh();
    }
}

impl SeenFingerprints {
    #[inline]
    fn new() -> Self {
        Self {
            inline: [EMPTY_FINGERPRINT; INLINE_SEEN_CAP],
            len: 0,
            spill: None,
        }
    }

    #[inline]
    fn insert(&mut self, fp: Fingerprint) -> bool {
        if let Some(spill) = &mut self.spill {
            return spill.insert(fp);
        }

        if fp == EMPTY_FINGERPRINT {
            // Collision with sentinel — go straight to spill set to avoid confusion
            let mut spill = FxHashSet::default();
            spill.reserve(INLINE_SEEN_CAP * 2);
            for existing in &self.inline {
                if *existing != EMPTY_FINGERPRINT {
                    spill.insert(*existing);
                }
            }
            let inserted = spill.insert(fp);
            self.spill = Some(spill);
            return inserted;
        }

        // Load factor ~75% to keep probing fast
        if self.len < (INLINE_SEEN_CAP * 3 / 4) {
            let mut idx = fingerprint_slot(fp);
            loop {
                if self.inline[idx] == EMPTY_FINGERPRINT {
                    self.inline[idx] = fp;
                    self.len += 1;
                    return true;
                }
                if self.inline[idx] == fp {
                    return false;
                }
                idx = (idx + 1) & (INLINE_SEEN_CAP - 1);
            }
        }

        let mut spill = FxHashSet::default();
        spill.reserve(INLINE_SEEN_CAP * 2);
        for existing in &self.inline {
            if *existing != EMPTY_FINGERPRINT {
                spill.insert(*existing);
            }
        }
        let inserted = spill.insert(fp);
        self.spill = Some(spill);
        inserted
    }
}

#[inline(always)]
fn fingerprint_slot(fp: Fingerprint) -> usize {
    let mixed = fp.hi ^ fp.lo.rotate_left(31);
    (mixed as usize) & (INLINE_SEEN_CAP - 1)
}

impl SearchAccumulator {
    fn new(limit: usize) -> Self {
        Self {
            total_hits: 0,
            limit,
            items: if limit == 0 { Vec::new() } else { Vec::with_capacity(limit) },
        }
    }

    #[inline]
    fn record(&mut self, index: usize, length: usize, result: CandidateResult) {
        self.total_hits += 1;

        if self.limit == 0 {
            self.items.push(make_search_hit(index, length, result));
            return;
        }

        if self.items.len() < self.limit {
            self.items.push(make_search_hit(index, length, result));
            return;
        }

        let min_index = self.min_index();
        if result.step > self.items[min_index].step {
            self.items[min_index] = make_search_hit(index, length, result);
        }
    }

    #[inline]
    fn merge(mut self, other: Self) -> Self {
        self.total_hits += other.total_hits;

        if self.limit == 0 {
            self.items.extend(other.items);
            return self;
        }

        for hit in other.items {
            self.consider_hit(hit);
        }

        self
    }

    #[inline]
    fn consider_hit(&mut self, hit: SearchHit) {
        if self.items.len() < self.limit {
            self.items.push(hit);
            return;
        }

        let min_index = self.min_index();
        if hit.step > self.items[min_index].step {
            self.items[min_index] = hit;
        }
    }

    #[inline]
    fn min_index(&self) -> usize {
        let mut min_index = 0usize;
        let mut min_step = self.items[0].step;

        for (idx, hit) in self.items.iter().enumerate().skip(1) {
            if hit.step < min_step {
                min_step = hit.step;
                min_index = idx;
            }
        }

        min_index
    }
}

#[inline(always)]
fn mix_hash(hash: u64, value: u64, mul: u64) -> u64 {
    (hash ^ value.wrapping_mul(mul)).rotate_left(27).wrapping_mul(mul)
}

#[inline]
fn group_from_items(mut items: Vec<Item>) -> Item {
    if items.len() == 1 {
        items.pop().unwrap()
    } else {
        Item::Group(Arc::new(Expr::new(items)))
    }
}

#[inline]
fn expand_item(item: &Item) -> Vec<Item> {
    match item {
        Item::Comb(c) => vec![Item::Comb(*c)],
        Item::Group(expr) => expr.items.clone(),
    }
}

#[inline]
fn append_expanded(item: &Item, out: &mut Vec<Item>) {
    match item {
        Item::Comb(c) => out.push(Item::Comb(*c)),
        Item::Group(expr) => out.extend(expr.items.iter().cloned()),
    }
}

#[inline]
fn expanded_twice(item: &Item) -> Vec<Item> {
    let item_len = match item {
        Item::Comb(_) => 1,
        Item::Group(expr) => expr.items.len(),
    };
    let mut items = Vec::with_capacity(item_len * 2);
    append_expanded(item, &mut items);
    append_expanded(item, &mut items);
    items
}

#[inline]
fn parse(s: &str) -> Expr {
    let mut stack: Vec<Vec<Item>> = vec![Vec::with_capacity(s.len())];

    for b in s.bytes() {
        match b {
            b'(' => stack.push(Vec::new()),
            b')' => {
                if stack.len() > 1 {
                    let items = stack.pop().unwrap();
                    stack.last_mut().unwrap().push(group_from_items(items));
                }
            }
            b if b.is_ascii_whitespace() => {}
            _ => stack.last_mut().unwrap().push(Item::Comb(b)),
        }
    }

    while stack.len() > 1 {
        let items = stack.pop().unwrap();
        stack.last_mut().unwrap().push(group_from_items(items));
    }

    Expr::new(stack.pop().unwrap())
}

#[inline]
fn push_item_string(item: &Item, out: &mut String) {
    match item {
        Item::Comb(c) => out.push(*c as char),
        Item::Group(expr) => {
            out.push('(');
            for child in &expr.items {
                push_item_string(child, out);
            }
            out.push(')');
        }
    }
}

#[inline]
fn push_item_string_limited(item: &Item, out: &mut String, limit: usize) -> bool {
    match item {
        Item::Comb(c) => {
            if out.len() == limit {
                return false;
            }
            out.push(*c as char);
            true
        }
        Item::Group(expr) => {
            if out.len() == limit {
                return false;
            }
            out.push('(');
            for child in &expr.items {
                if !push_item_string_limited(child, out, limit) {
                    return false;
                }
            }
            if out.len() == limit {
                return false;
            }
            out.push(')');
            true
        }
    }
}

#[inline]
fn stringify(expr: &Expr) -> String {
    let mut out = String::with_capacity(expr.flat_len + expr.items.len());
    for item in &expr.items {
        push_item_string(item, &mut out);
    }
    out
}

#[inline]
fn stringify_prefix(expr: &Expr, limit: usize) -> String {
    let mut out = String::with_capacity(limit.min(expr.flat_len + expr.items.len()));
    for item in &expr.items {
        if !push_item_string_limited(item, &mut out, limit) {
            break;
        }
    }
    out
}

#[inline]
fn expr_len(expr: &Expr) -> usize {
    expr.flat_len
}

#[inline]
fn apply_rule(expr: &mut Expr, pos: usize, c: u8, step_budget: usize) -> usize {
    let rest_len = expr.items.len() - pos - 1;

    match c {
        b'R' if rest_len >= 1 => {
            let x = expr.items[pos + 1].clone();
            expr.items[pos] = x.clone();
            expr.items[pos + 1] = x;
            expr.refresh();
            1
        }
        b'E' if rest_len >= 2 => {
            let x = expr.items[pos + 1].clone();
            expr.replace_consumed(pos, 3, [x]);
            1
        }
        b'S' if rest_len >= 2 => {
            let x = expr.items[pos + 1].clone();
            let y = expr.items[pos + 2].clone();

            let xx = group_from_items(expanded_twice(&x));
            let yy = group_from_items(expanded_twice(&y));

            expr.replace_consumed(pos, 3, [xx.clone(), yy, xx]);
            1
        }
        b'O' if rest_len >= 2 => {
            let x = expr.items[pos + 1].clone();
            let y = expr.items[pos + 2].clone();

            let y_len = match &y {
                Item::Comb(_) => 1,
                Item::Group(inner) => inner.items.len(),
            };
            let mut yx_items = Vec::with_capacity(y_len + 1);
            append_expanded(&y, &mut yx_items);
            yx_items.push(x.clone());
            let yx = group_from_items(yx_items);

            expr.replace_consumed(pos, 3, [x.clone(), yx, x, y]);
            1
        }
        b'I' if rest_len >= 3 => {
            let x = expr.items[pos + 1].clone();
            let y = expr.items[pos + 2].clone();
            let z = expr.items[pos + 3].clone();

            let x_len = match &x {
                Item::Comb(_) => 1,
                Item::Group(inner) => inner.items.len(),
            };
            let mut xz_items = Vec::with_capacity(x_len + 1);
            append_expanded(&x, &mut xz_items);
            xz_items.push(z);
            let xz = group_from_items(xz_items);

            expr.replace_consumed(pos, 4, [y, xz]);
            1
        }
        b'D' if rest_len >= 3 => {
            let x = expr.items[pos + 1].clone();
            let y = expr.items[pos + 2].clone();
            let z = expr.items[pos + 3].clone();

            let y_len = match &y {
                Item::Comb(_) => 1,
                Item::Group(inner) => inner.items.len(),
            };
            let mut yyz_items = Vec::with_capacity(y_len * 2 + 1);
            append_expanded(&y, &mut yyz_items);
            append_expanded(&y, &mut yyz_items);
            yyz_items.push(z);
            let yyz = group_from_items(yyz_items);

            expr.replace_consumed(pos, 4, [x.clone(), x, yyz]);
            1
        }
        b'T' if rest_len >= 2 => {
            let a = expr.items[pos + 1].clone();
            expr.items.drain(pos..pos + 2);
            expr.items.push(a);
            expr.refresh();
            1
        }
        b'C' if rest_len >= 1 => {
            // Macro fusion: CR x ...rest
            // Step 1 (C fires): R x ...rest R   (C clones R, puts copy at end)
            // Step 2 (R fires): x x ...rest R   (R duplicates x)
            if step_budget >= 2 && rest_len >= 2 && matches!(expr.items[pos + 1], Item::Comb(b'R')) {
                let x = expr.items[pos + 2].clone();
                // Remove C, R, x  and replace with x, x
                expr.items.splice(pos..pos + 3, [x.clone(), x]);
                // R goes to the END of the entire expression
                expr.items.push(Item::Comb(b'R'));
                expr.refresh();
                return 2;
            }
            let a = expr.items[pos + 1].clone();
            expr.items.remove(pos);
            expr.items.push(a);
            expr.refresh();
            1
        }
        b'N' if rest_len >= 1 => {
            expr.items.remove(pos);
            expr.items[pos..].reverse();
            expr.refresh();
            1
        }
        _ => 0,
    }
}

#[inline]
fn has_rule_at(expr: &Expr, pos: usize, c: u8) -> bool {
    let rest_len = expr.items.len() - pos - 1;
    match c {
        b'R' | b'C' | b'N' => rest_len >= 1,
        b'E' | b'S' | b'O' | b'T' => rest_len >= 2,
        b'I' | b'D' => rest_len >= 3,
        _ => false,
    }
}

fn has_redex(expr: &Expr) -> bool {
    for (i, item) in expr.items.iter().enumerate() {
        if let Item::Comb(c) = item {
            if has_rule_at(expr, i, *c) {
                return true;
            }
        }
    }

    expr.items.iter().any(|item| match item {
        Item::Comb(_) => false,
        Item::Group(inner) => has_redex(inner),
    })
}

fn reduce_once(expr: &mut Expr, step_budget: usize) -> usize {
    let top_len = expr.items.len();
    let mut i = 0usize;

    while i < top_len {
        let c = match &expr.items[i] {
            Item::Comb(c) => *c,
            Item::Group(_) => {
                i += 1;
                continue;
            }
        };

        let steps = apply_rule(expr, i, c, step_budget);
        if steps != 0 {
            return steps;
        }

        i += 1;
    }

    let child_len = expr.items.len();
    let mut i = 0usize;

    while i < child_len {
        let mut replacement = None;

        match &mut expr.items[i] {
            Item::Comb(_) => {}
            Item::Group(inner) => {
                if Arc::strong_count(inner) != 1 && !has_redex(inner) {
                    i += 1;
                    continue;
                }
                let inner = Arc::make_mut(inner);
                let steps = reduce_once(inner, step_budget);
                if steps != 0 {
                    if inner.items.len() == 1 {
                        replacement = inner.items.pop();
                    }
                    if let Some(item) = replacement {
                        expr.items[i] = item;
                    }
                    expr.refresh();
                    return steps;
                }
            }
        }

        i += 1;
    }

    0
}

fn render_truncated(expr: &Expr) -> String {
    format!("{}... (truncated)", stringify_prefix(expr, TRUNCATED_RENDER_LEN))
}

#[inline]
fn should_checkpoint_cycle(prev_len: usize, next_len: usize, stable_steps: &mut usize) -> bool {
    if next_len < prev_len {
        *stable_steps = 0;
        return true;
    }

    if next_len == prev_len {
        *stable_steps += 1;
        return *stable_steps >= STABLE_CYCLE_CHECK_STEPS;
    }

    *stable_steps = 0;
    false
}

fn reduce_full(s: &str, max_steps: usize, max_len: usize) -> ReduceResult {
    let mut expr = parse(s);
    let mut seen = SeenFingerprints::new();
    seen.insert(expr.fingerprint());

    let mut step = 0usize;
    let mut stable_steps = 0usize;
    while step < max_steps {
        let prev = expr.fingerprint();
        let prev_len = expr.flat_len;

        let step_delta = reduce_once(&mut expr, max_steps - step);
        if step_delta == 0 {
            return ReduceResult { result: stringify(&expr), steps: step, status: "normal" };
        }

        let next = expr.fingerprint();
        if next == prev {
            return ReduceResult { result: stringify(&expr), steps: step, status: "fixed_point" };
        }

        if expr_len(&expr) > max_len {
            return ReduceResult { result: render_truncated(&expr), steps: step, status: "limit_reached" };
        }

        step = (step + step_delta).min(max_steps);
        if should_checkpoint_cycle(prev_len, expr.flat_len, &mut stable_steps) && !seen.insert(next) {
            return ReduceResult { result: stringify(&expr), steps: step, status: "cycle" };
        }
    }

    ReduceResult { result: stringify(&expr), steps: step, status: "limit_reached" }
}

fn reduce_search_candidate(index: usize, length: usize, max_steps: usize, max_len: usize) -> CandidateResult {
    let mut expr = Expr::from_index(index, length);
    let mut seen = SeenFingerprints::new();
    seen.insert(expr.fingerprint());

    let mut step = 0usize;
    let mut min_len_this_epoch = expr.flat_len;
    let mut epochs = [usize::MAX; 3];
    let mut epoch_count = 0usize;
    let mut stable_steps = 0usize;

    while step < max_steps {
        if expr.flat_len < min_len_this_epoch {
            min_len_this_epoch = expr.flat_len;
        }

        if step > 0 && step % 100 == 0 {
            epochs[epoch_count % 3] = min_len_this_epoch;
            epoch_count += 1;
            min_len_this_epoch = usize::MAX;

            if epoch_count >= 3 {
                let a = epochs[(epoch_count - 3) % 3];
                let b = epochs[(epoch_count - 2) % 3];
                let c = epochs[(epoch_count - 1) % 3];
                if a < b && b < c {
                    return CandidateResult { final_expr: expr, step, status: "divergent" };
                }
            }
        }

        let prev = expr.fingerprint();
        let prev_len = expr.flat_len;
        let step_delta = reduce_once(&mut expr, max_steps - step);
        if step_delta == 0 {
            return CandidateResult { final_expr: expr, step, status: "normal" };
        }

        let next = expr.fingerprint();
        if next == prev {
            return CandidateResult { final_expr: expr, step, status: "fixed_point" };
        }

        if expr.flat_len > max_len {
            return CandidateResult { final_expr: expr, step, status: "limit_reached" };
        }

        step = (step + step_delta).min(max_steps);
        if should_checkpoint_cycle(prev_len, expr.flat_len, &mut stable_steps) && !seen.insert(next) {
            return CandidateResult { final_expr: expr, step, status: "cycle" };
        }
    }

    CandidateResult { final_expr: expr, step, status: "limit_reached" }
}

#[inline]
fn index_to_string(mut index: usize, length: usize) -> String {
    let mut bytes = vec![ALPHABET[0]; length];
    for slot in bytes.iter_mut().rev() {
        *slot = ALPHABET[index % ALPHABET.len()];
        index /= ALPHABET.len();
    }
    String::from_utf8(bytes).unwrap()
}

#[inline]
fn make_search_hit(index: usize, length: usize, result: CandidateResult) -> SearchHit {
    SearchHit {
        expr: index_to_string(index, length),
        out: stringify_prefix(&result.final_expr, SEARCH_PREVIEW_LEN + 1),
        step: result.step,
        status: result.status,
    }
}

fn find_s_matches_in_domain(
    length: usize,
    max_steps: usize,
    max_len: usize,
    threads: Option<usize>,
) -> Vec<SMatch> {
    let target = "XZ(YZ)";
    let total = ALPHABET.len().pow(length as u32);

    let job = || {
        (0..total)
            .into_par_iter()
            .filter_map(|index| {
                let expr = index_to_string(index, length);
                let lhs = format!("{}XYZ", expr);
                let res = reduce_full(&lhs, max_steps, max_len);

                if res.status == "limit_reached" || res.status == "cycle" {
                    return None;
                }

                if res.result == target {
                    return Some(SMatch {
                        expr,
                        steps: res.steps,
                        status: res.status,
                    });
                }

                None
            })
            .collect::<Vec<SMatch>>()
    };

    let mut matches = if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
            .install(job)
    } else {
        job()
    };

    matches.sort_by(|a, b| a.steps.cmp(&b.steps).then_with(|| a.expr.cmp(&b.expr)));
    matches
}

#[inline]
fn fill_placeholders(template: &str, candidate: &str) -> Option<String> {
    if !template.bytes().any(|b| b == b'.' || b == b'?') {
        return None;
    }

    let mut out = String::with_capacity(template.len() + candidate.len());
    for ch in template.chars() {
        if ch == '.' || ch == '?' {
            out.push_str(candidate);
        } else if !ch.is_whitespace() {
            out.push(ch);
        }
    }
    Some(out)
}

#[inline]
fn normalize_equation_side(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_whitespace() {
            continue;
        }
        out.push(ch.to_ascii_uppercase());
    }
    out
}

fn find_equation_matches_in_domain(
    lhs_template: &str,
    rhs_target: &str,
    length: usize,
    max_steps: usize,
    max_len: usize,
    threads: Option<usize>,
) -> Vec<SMatch> {
    let total = ALPHABET.len().pow(length as u32);

    let job = || {
        (0..total)
            .into_par_iter()
            .filter_map(|index| {
                let expr = index_to_string(index, length);
                let lhs = fill_placeholders(lhs_template, &expr)?;
                let res = reduce_full(&lhs, max_steps, max_len);

                if res.status == "limit_reached" || res.status == "cycle" {
                    return None;
                }
                if res.result != rhs_target {
                    return None;
                }

                Some(SMatch {
                    expr,
                    steps: res.steps,
                    status: res.status,
                })
            })
            .collect::<Vec<SMatch>>()
    };

    let mut matches = if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
            .install(job)
    } else {
        job()
    };

    matches.sort_by(|a, b| a.steps.cmp(&b.steps).then_with(|| a.expr.cmp(&b.expr)));
    matches
}

#[derive(Parser)]
#[command(name = "stoicn", about = "STROIED-CN Interpreter & Toolkit (Rust engine)", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a .stoicn file
    Run {
        file: String,
        #[arg(long, default_value_t = 500)]
        max_steps: usize,
        #[arg(long, default_value_t = DEFAULT_MAX_LEN)]
        max_len: usize,
    },
    /// Evaluate a single expression
    Eval {
        expression: Vec<String>,
        #[arg(long, default_value_t = 500)]
        max_steps: usize,
        #[arg(long, default_value_t = DEFAULT_MAX_LEN)]
        max_len: usize,
    },
    /// Interactive REPL
    Repl {
        #[arg(long, default_value_t = 500)]
        max_steps: usize,
        #[arg(long, default_value_t = DEFAULT_MAX_LEN)]
        max_len: usize,
    },
    /// Enumerate all strings of given length
    Search {
        length: usize,
        #[arg(long, default_value_t = 50)]
        max_steps: usize,
        #[arg(long, default_value_t = DEFAULT_MAX_LEN)]
        max_len: usize,
        #[arg(long)]
        filter: Option<String>,
        #[arg(long, alias = "set-LBlength", default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Run benchmark tests
    Bench,
    /// Search ST(1..=domain) for combinator strings that behave like SKI S.
    /// Examples:
    ///   stoicn find S 5
    ///   stoicn find .XYZ = XZ(YZ) 5
    Find {
        query: Vec<String>,
        #[arg(long, default_value_t = 300)]
        max_steps: usize,
        #[arg(long, default_value_t = DEFAULT_MAX_LEN)]
        max_len: usize,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 5)]
        show: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Eval { expression, max_steps, max_len } => {
            let expr = expression.join("");
            println!("======================================================");
            println!("  STROIED-CN Interpreter (Rust)");
            println!("======================================================");
            println!("  Expression: {}\n", expr);
            let start = Instant::now();
            let res = reduce_full(&expr, *max_steps, *max_len);
            let elapsed = start.elapsed();
            println!("  Result: {}", res.result);
            println!("  Steps: {}, Status: {}", res.steps, res.status);
            println!("  Time: {:.4}s\n", elapsed.as_secs_f64());
        }
        Commands::Run { file, max_steps, max_len } => {
            let content = fs::read_to_string(file).expect("Failed to read file");
            println!("======================================================");
            println!("  STROIED-CN Interpreter (Rust)");
            println!("======================================================");
            for (i, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                println!("> Line {}: {}", i + 1, line);
                let start = Instant::now();
                let res = reduce_full(line, *max_steps, *max_len);
                let elapsed = start.elapsed();
                println!("  Result: {}", res.result);
                println!("  Steps: {} | Time: {:.4}s\n", res.steps, elapsed.as_secs_f64());
            }
        }
        Commands::Repl { max_steps, max_len } => {
            println!("======================================================");
            println!("  STROIED-CN REPL (Rust)");
            println!("======================================================");
            println!("  Type an expression and press Enter.");
            println!("  Commands: :quit  :help\n");

            loop {
                print!("stoicn> ");
                io::stdout().flush().unwrap();

                let mut input = String::new();
                if io::stdin().read_line(&mut input).is_err() || input.trim().is_empty() {
                    continue;
                }

                let input = input.trim();
                if input == ":quit" {
                    break;
                }
                if input == ":help" {
                    println!("S T R O I E D C N");
                    continue;
                }

                let start = Instant::now();
                let mut expr = parse(input);
                let mut seen = SeenFingerprints::new();
                seen.insert(expr.fingerprint());

                let mut step = 0usize;
                let mut status = "limit_reached";
                let mut stable_steps = 0usize;

                while step < *max_steps {
                    let prev = expr.fingerprint();
                    let prev_len = expr.flat_len;

                    let step_delta = reduce_once(&mut expr, *max_steps - step);
                    if step_delta == 0 {
                        status = "normal";
                        break;
                    }

                    let next = expr.fingerprint();
                    if next == prev {
                        status = "fixed_point";
                        break;
                    }

                    if expr_len(&expr) > *max_len {
                        status = "limit_reached";
                        break;
                    }

                    step = (step + step_delta).min(*max_steps);
                    if should_checkpoint_cycle(prev_len, expr.flat_len, &mut stable_steps) && !seen.insert(next) {
                        status = "cycle";
                        break;
                    }
                }

                let elapsed = start.elapsed();
                println!("  Result: {}", stringify(&expr));
                println!("  Steps: {}, Status: {}, Time: {:.4}s\n", step, status, elapsed.as_secs_f64());
            }
        }
        Commands::Search { length, max_steps, max_len, filter, limit, threads } => {
            println!("======================================================");
            println!("  STROIED-CN Search (length={})", length);
            println!("======================================================");

            let total = ALPHABET.len().pow(*length as u32);
            println!("  Total expressions: {}", total);

            let start = Instant::now();
            let filter = filter.as_deref();
            let keep_limit = *limit;

            let search_job = || {
                (0..total)
                    .into_par_iter()
                    .fold(
                        || SearchAccumulator::new(keep_limit),
                        |mut acc, index| {
                            let result = reduce_search_candidate(index, *length, *max_steps, *max_len);
                            let keep = match filter {
                                Some(wanted) => wanted == result.status,
                                None => true,
                            };

                            if keep {
                                acc.record(index, *length, result);
                            }

                            acc
                        },
                    )
                    .reduce(
                        || SearchAccumulator::new(keep_limit),
                        |left, right| left.merge(right),
                    )
            };

            let mut hits = if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(*t)
                    .build()
                    .unwrap()
                    .install(search_job)
            } else {
                search_job()
            };

            hits.items.sort_by(|a, b| {
                b.step
                    .cmp(&a.step)
                    .then_with(|| a.expr.cmp(&b.expr))
                    .then_with(|| a.status.cmp(b.status))
            });

            let display_limit = if *limit == 0 { hits.items.len() } else { *limit };
            for hit in hits.items.iter().take(display_limit) {
                let display_out = if hit.out.len() > 40 {
                    format!("{}...", &hit.out[..37])
                } else {
                    hit.out.clone()
                };
                println!("  {:12} {:40} {:5}  {}", hit.expr, display_out, hit.step, hit.status);
            }

            let elapsed = start.elapsed();
            println!(
                "\n  Found {} matches in {:.4}s ({} exprs/sec)",
                hits.total_hits,
                elapsed.as_secs_f64(),
                (total as f64 / elapsed.as_secs_f64()) as u64
            );
        }
        Commands::Bench => {
            println!("Running benchmarks...");
            let tests = [
                ("DDDDD", 500),
                ("DDDDDD", 500),
                ("SDSD", 500),
                ("SDSDS", 500),
                ("DSDSD", 500),
                ("OSOSOS", 200),
                ("SSSSSSSS", 200),
                ("DSDSDSDS", 200),
                ("SODSODSOD", 100),
            ];

            for (expr, limit) in tests {
                let start = Instant::now();
                let res = reduce_full(expr, limit, DEFAULT_MAX_LEN);
                let elapsed = start.elapsed();
                println!(
                    "{:15} ms={:4}, steps={:4}, status={:14}, len={:8}, time={:.4}s",
                    expr,
                    limit,
                    res.steps,
                    res.status,
                    res.result.len(),
                    elapsed.as_secs_f64()
                );
            }
        }
        Commands::Find { query, max_steps, max_len, threads, show } => {
            if query.len() < 2 {
                eprintln!("Usage: stoicn find S <domain> OR stoicn find <lhs> = <rhs> <domain>");
                return;
            }

            let domain = match query.last().and_then(|s| s.parse::<usize>().ok()) {
                Some(d) => d,
                None => {
                    eprintln!("Last argument must be a domain number.");
                    return;
                }
            };
            if domain == 0 {
                eprintln!("Domain must be >= 1.");
                return;
            }

            let parts = &query[..query.len() - 1];
            let mut mode_s = false;
            let mut lhs_template = String::new();
            let mut rhs_target = String::new();
            let mut auto_xyz = false;

            if parts.len() == 1 && parts[0].eq_ignore_ascii_case("S") {
                mode_s = true;
            } else {
                let Some(eq_pos) = parts.iter().position(|p| p == "=") else {
                    eprintln!("Equation mode requires '='. Example: stoicn find .XYZ = XZ(YZ) 5");
                    return;
                };
                if eq_pos == 0 || eq_pos + 1 >= parts.len() {
                    eprintln!("Equation must include both LHS and RHS.");
                    return;
                }
                lhs_template = normalize_equation_side(&parts[..eq_pos].join(""));
                rhs_target = normalize_equation_side(&parts[eq_pos + 1..].join(""));
                if !lhs_template.contains('.') && !lhs_template.contains('?') {
                    eprintln!("LHS must contain '.' or '?' placeholder(s).");
                    return;
                }

                let has_vars = lhs_template.contains('X') || lhs_template.contains('Y') || lhs_template.contains('Z');
                if !has_vars {
                    lhs_template.push_str("XYZ");
                    auto_xyz = true;
                }
            }

            println!("======================================================");
            println!("  STROIED-CN Find S (domains 1..={})", domain);
            println!("======================================================");
            if mode_s {
                println!("  Equation checked: ?XYZ => XZ(YZ)");
            } else {
                println!("  Equation checked: {} => {}", lhs_template, rhs_target);
                if auto_xyz {
                    println!("  Note: no XYZ variables in LHS; treated as {}.", lhs_template);
                }
            }

            for d in 1..=domain {
                let start = Instant::now();
                let matches = if mode_s {
                    find_s_matches_in_domain(d, *max_steps, *max_len, *threads)
                } else {
                    find_equation_matches_in_domain(
                        &lhs_template,
                        &rhs_target,
                        d,
                        *max_steps,
                        *max_len,
                        *threads,
                    )
                };
                let elapsed = start.elapsed();

                if matches.is_empty() {
                    println!(
                        "  ST({:2}): no S-like combinator found in {:.4}s",
                        d,
                        elapsed.as_secs_f64()
                    );
                    continue;
                }

                println!(
                    "  ST({:2}): found {} match(es) in {:.4}s",
                    d,
                    matches.len(),
                    elapsed.as_secs_f64()
                );

                for m in matches.iter().take(*show) {
                    println!("    {:12} steps={:4} status={}", m.expr, m.steps, m.status);
                }
            }
        }
    }
}
