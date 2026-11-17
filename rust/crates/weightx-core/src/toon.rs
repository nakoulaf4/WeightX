// SPDX-License-Identifier: GPL-3.0-or-later
//! TOON (Token-Oriented Object Notation) serialization, parsing, and token diagnostics.
//!
//! Rust port of `weightx.toon_io` with a full built-in encoder AND decoder
//! (the Python fallback relied on PyYAML for decoding; here TOON round-trips natively).

use regex::Regex;
use serde_json::{json, Map, Value};

/// Round floats to 6 decimals; map non-finite to null (mirrors `_sanitize_for_json`).
pub fn sanitize(v: &Value) -> Value {
    match v {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    Value::Null
                } else if n.is_i64() || n.is_u64() {
                    v.clone()
                } else {
                    let r = (f * 1e6).round() / 1e6;
                    json!(r)
                }
            } else {
                v.clone()
            }
        }
        Value::Array(a) => Value::Array(a.iter().map(sanitize).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, val)| (k.clone(), sanitize(val)))
                .collect::<Map<String, Value>>(),
        ),
        other => other.clone(),
    }
}

/// C/Python `%.6g`-style formatting.
pub fn fmt_g(v: f64, sig: usize) -> String {
    if !v.is_finite() {
        return "null".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let exp = v.abs().log10().floor() as i32;
    if exp < -4 || exp >= sig as i32 {
        let s = format!("{:.*e}", sig.saturating_sub(1), v);
        // Rust: "1.2345e2" -> trim mantissa zeros, normalize exponent to e+XX
        let (mant, e) = s.split_once('e').unwrap();
        let mant = trim_zeros(mant);
        let ev: i32 = e.parse().unwrap();
        format!("{}e{}{:02}", mant, if ev < 0 { '-' } else { '+' }, ev.abs())
    } else {
        let decimals = (sig as i32 - 1 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, v);
        trim_zeros(&s).to_string()
    }
}

fn trim_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

fn is_primitive(v: &Value) -> bool {
    !v.is_object() && !v.is_array()
}

fn fmt_prim(v: &Value, delimiter: &str) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                let f = n.as_f64().unwrap_or(0.0);
                if !f.is_finite() {
                    return "null".to_string();
                }
                // Shortest round-trip float (matches the `toon_format` reference
                // encoder used by Python); %.6g here would lose precision.
                serde_json::to_string(&Value::from(f)).unwrap()
            }
        }
        Value::String(s) => {
            if s.is_empty()
                || s.contains(delimiter)
                || s.contains('\n')
                || s.contains(':')
                || s.trim() != s
                || matches!(s.as_str(), "null" | "true" | "false")
                || s.parse::<f64>().is_ok()
            {
                serde_json::to_string(s).unwrap()
            } else {
                s.clone()
            }
        }
        _ => String::new(),
    }
}

struct Encoder {
    lines: Vec<String>,
    unit: String,
    delimiter: String,
}

impl Encoder {
    fn emit(&mut self, depth: usize, text: &str) {
        self.lines.push(format!("{}{}", self.unit.repeat(depth), text));
    }

    fn render_field(&mut self, key: &str, val: &Value, depth: usize) {
        match val {
            Value::Object(m) => {
                self.emit(depth, &format!("{}:", key));
                for (k, v) in m {
                    self.render_field(k, v, depth + 1);
                }
            }
            Value::Array(a) => self.render_array(key, a, depth),
            other => self.emit(depth, &format!("{}: {}", key, fmt_prim(other, &self.delimiter))),
        }
    }

    fn render_array(&mut self, prefix: &str, arr: &[Value], depth: usize) {
        if arr.is_empty() {
            let text = if prefix.is_empty() {
                "[]".to_string()
            } else {
                format!("{}: []", prefix)
            };
            self.emit(depth, &text);
            return;
        }
        if arr.iter().all(is_primitive) {
            let joined = arr
                .iter()
                .map(|x| fmt_prim(x, &self.delimiter))
                .collect::<Vec<_>>()
                .join(&self.delimiter);
            self.emit(depth, &format!("{}[{}]: {}", prefix, arr.len(), joined));
            return;
        }
        // Uniform tabular objects
        if arr.iter().all(|x| x.is_object() && !x.as_object().unwrap().is_empty()) {
            let keys: Vec<String> = arr[0].as_object().unwrap().keys().cloned().collect();
            let uniform = arr.iter().all(|x| {
                let m = x.as_object().unwrap();
                m.keys().eq(keys.iter()) && keys.iter().all(|k| is_primitive(&m[k]))
            });
            if uniform {
                self.emit(
                    depth,
                    &format!("{}[{}]{{{}}}:", prefix, arr.len(), keys.join(&self.delimiter)),
                );
                for row in arr {
                    let m = row.as_object().unwrap();
                    let line = keys
                        .iter()
                        .map(|k| fmt_prim(&m[k], &self.delimiter))
                        .collect::<Vec<_>>()
                        .join(&self.delimiter);
                    self.emit(depth + 1, &line);
                }
                return;
            }
        }
        self.emit(depth, &format!("{}[{}]:", prefix, arr.len()));
        for item in arr {
            match item {
                Value::Object(m) => {
                    let first_idx = self.lines.len();
                    for (k, v) in m {
                        self.render_field(k, v, depth + 2);
                    }
                    if self.lines.len() > first_idx {
                        let stripped = self.lines[first_idx].trim_start().to_string();
                        self.lines[first_idx] =
                            format!("{}- {}", self.unit.repeat(depth + 1), stripped);
                    }
                }
                other => self.emit(depth + 1, &format!("- {}", fmt_prim(other, &self.delimiter))),
            }
        }
    }
}

/// Serialize a JSON value into compact TOON.
pub fn dumps_toon(obj: &Value) -> String {
    let clean = sanitize(obj);
    let mut enc = Encoder {
        lines: Vec::new(),
        unit: "  ".to_string(),
        delimiter: ",".to_string(),
    };
    match &clean {
        Value::Object(m) => {
            for (k, v) in m {
                enc.render_field(k, v, 0);
            }
        }
        Value::Array(a) => enc.render_array("", a, 0),
        other => enc.emit(0, &fmt_prim(other, ",")),
    }
    enc.lines.join("\n")
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

fn parse_prim(s: &str) -> Value {
    let t = s.trim();
    match t {
        "null" => Value::Null,
        "true" => json!(true),
        "false" => json!(false),
        _ => {
            if t.starts_with('"') {
                serde_json::from_str(t).unwrap_or_else(|_| json!(t))
            } else if let Ok(i) = t.parse::<i64>() {
                json!(i)
            } else if let Ok(f) = t.parse::<f64>() {
                if t.contains('.') || t.contains('e') || t.contains('E') {
                    json!(f)
                } else {
                    json!(t)
                }
            } else {
                json!(t)
            }
        }
    }
}

/// Split on delimiter outside JSON-quoted strings.
fn split_values(s: &str, delim: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_str = false;
    let mut esc = false;
    for ch in s.chars() {
        if in_str {
            cur.push(ch);
            if esc {
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
        } else if ch == '"' {
            in_str = true;
            cur.push(ch);
        } else if ch == delim {
            out.push(cur.trim().to_string());
            cur = String::new();
        } else {
            cur.push(ch);
        }
    }
    out.push(cur.trim().to_string());
    out
}

struct Line {
    indent: usize,
    text: String,
}

fn lex_lines(text: &str) -> Vec<Line> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let indent = l.len() - l.trim_start().len();
            Line {
                indent,
                text: l.trim().to_string(),
            }
        })
        .collect()
}

struct Decoder<'a> {
    lines: &'a [Line],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn peek(&self) -> Option<&'a Line> {
        self.lines.get(self.pos)
    }

    /// Parse a block of fields at `indent`. Returns object.
    fn parse_map(&mut self, indent: usize) -> Value {
        let mut m = Map::new();
        while let Some(line) = self.peek() {
            if line.indent < indent || line.text.starts_with("- ") {
                break;
            }
            if line.indent > indent {
                // stray deeper line; skip defensively
                self.pos += 1;
                continue;
            }
            self.pos += 1;
            let (key, rest) = split_key(&line.text);
            let val = self.parse_value_after_key(key, rest, indent);
            m.insert(base_key(key).to_string(), val);
        }
        Value::Object(m)
    }

    fn parse_value_after_key(&mut self, key: &str, rest: &str, indent: usize) -> Value {
        // Array header?  key[N]... or key[N]{cols}:
        if let Some(br) = key.find('[') {
            if key.ends_with(']') || key.contains("]{") {
                return self.parse_array_header(key, rest, indent);
            }
        }
        if rest.is_empty() {
            // nested map or empty
            if let Some(next) = self.peek() {
                if next.indent > indent && !next.text.starts_with("- ") {
                    return self.parse_map(next.indent);
                }
            }
            return Value::Object(Map::new());
        }
        parse_prim(rest)
    }

    fn parse_array_header(&mut self, key: &str, rest: &str, indent: usize) -> Value {
        let br = key.find('[').unwrap();
        let close = key.find(']').unwrap();
        let n: usize = key[br + 1..close].parse().unwrap_or(0);
        let after = &key[close + 1..];
        // tabular {cols}
        if after.starts_with('{') && after.ends_with('}') {
            let cols = split_values(&after[1..after.len() - 1], ',');
            let mut rows = Vec::new();
            for _ in 0..n {
                if let Some(line) = self.peek() {
                    if line.indent > indent {
                        self.pos += 1;
                        let vals = split_values(&line.text, ',');
                        let mut m = Map::new();
                        for (i, c) in cols.iter().enumerate() {
                            m.insert(
                                c.clone(),
                                vals.get(i).map(|v| parse_prim(v)).unwrap_or(Value::Null),
                            );
                        }
                        rows.push(Value::Object(m));
                        continue;
                    }
                }
                break;
            }
            return Value::Array(rows);
        }
        // inline primitives: rest non-empty
        if !rest.is_empty() {
            let vals = split_values(rest, ',');
            return Value::Array(vals.iter().map(|v| parse_prim(v)).collect());
        }
        if rest == "[]" {
            return Value::Array(vec![]);
        }
        // block list items
        let mut items = Vec::new();
        while let Some(line) = self.peek() {
            if line.indent <= indent || !line.text.starts_with("- ") {
                break;
            }
            let item_indent = line.indent;
            self.pos += 1;
            let content = line.text[2..].to_string();
            if content.contains(": ") || content.ends_with(':') {
                // object item: first field inline, remaining fields deeper
                let mut m = Map::new();
                let (k, r) = split_key(&content);
                let v = self.parse_value_after_key(k, r, item_indent + 2);
                m.insert(base_key(k).to_string(), v);
                while let Some(next) = self.peek() {
                    if next.indent > item_indent && !next.text.starts_with("- ") {
                        let sub_indent = next.indent;
                        let extra = self.parse_map(sub_indent);
                        if let Value::Object(em) = extra {
                            for (k2, v2) in em {
                                m.insert(k2, v2);
                            }
                        }
                    } else {
                        break;
                    }
                }
                items.push(Value::Object(m));
            } else {
                items.push(parse_prim(&content));
            }
            if items.len() >= n && n > 0 {
                // keep consuming only while items remain (n is authoritative)
                if items.len() == n {
                    break;
                }
            }
        }
        Value::Array(items)
    }
}

fn base_key(key: &str) -> &str {
    match key.find('[') {
        Some(i) => &key[..i],
        None => key,
    }
}

fn split_key(text: &str) -> (&str, &str) {
    // find first ": " or trailing ':' outside quotes/brackets
    let bytes = text.as_bytes();
    let mut in_str = false;
    for i in 0..bytes.len() {
        let c = bytes[i] as char;
        if c == '"' {
            in_str = !in_str;
        }
        if !in_str && c == ':' {
            if i + 1 == bytes.len() {
                return (&text[..i], "");
            }
            if bytes[i + 1] == b' ' {
                return (&text[..i], text[i + 2..].trim());
            }
        }
    }
    (text, "")
}

/// Deserialize a TOON document back into JSON values.
pub fn loads_toon(text: &str) -> Value {
    let lines = lex_lines(text);
    if lines.is_empty() {
        return Value::Null;
    }
    let mut dec = Decoder { lines: &lines, pos: 0 };
    let first = &lines[0];
    if first.text.starts_with('[') {
        // top-level array
        let (key, rest) = ("", first.text.as_str());
        let _ = key;
        dec.pos += 1;
        return dec.parse_array_header(&first.text[..first.text.find(':').unwrap_or(first.text.len())], rest_after_colon(&first.text), 0);
    }
    dec.parse_map(first.indent)
}

fn rest_after_colon(text: &str) -> &str {
    match text.find(": ") {
        Some(i) => text[i + 2..].trim(),
        None => "",
    }
}

// ---------------------------------------------------------------------------
// Token diagnostics
// ---------------------------------------------------------------------------

pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let re = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?|[^\sA-Za-z0-9_]").unwrap();
    let mut count = 0usize;
    for m in re.find_iter(text) {
        let tok = m.as_str();
        let first_alpha = tok.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false);
        if tok.len() > 8 && first_alpha {
            count += (tok.len() + 4) / 5;
        } else {
            count += 1;
        }
    }
    count.max(1)
}

#[derive(Debug, Clone)]
pub struct TokenSavingsReport {
    pub json_chars: usize,
    pub toon_chars: usize,
    pub json_est_tokens: usize,
    pub toon_est_tokens: usize,
    pub char_reduction_ratio: f64,
    pub token_reduction_ratio: f64,
}

impl TokenSavingsReport {
    pub fn to_value(&self) -> Value {
        json!({
            "json_chars": self.json_chars,
            "toon_chars": self.toon_chars,
            "json_est_tokens": self.json_est_tokens,
            "toon_est_tokens": self.toon_est_tokens,
            "char_reduction_pct": (self.char_reduction_ratio * 10000.0).round() / 100.0,
            "token_reduction_pct": (self.token_reduction_ratio * 10000.0).round() / 100.0,
        })
    }
}

pub fn compare_toon_vs_json(obj: &Value) -> TokenSavingsReport {
    let clean = sanitize(obj);
    let json_str = serde_json::to_string_pretty(&clean).unwrap();
    let toon_str = dumps_toon(&clean);
    let j_chars = json_str.chars().count();
    let t_chars = toon_str.chars().count();
    let j_toks = estimate_tokens(&json_str);
    let t_toks = estimate_tokens(&toon_str);
    TokenSavingsReport {
        json_chars: j_chars,
        toon_chars: t_chars,
        json_est_tokens: j_toks,
        toon_est_tokens: t_toks,
        char_reduction_ratio: (j_chars as f64 - t_chars as f64) / (j_chars.max(1)) as f64,
        token_reduction_ratio: (j_toks as f64 - t_toks as f64) / (j_toks.max(1)) as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_tabular() {
        let v = json!({
            "tensors": [
                {"name": "a.weight", "shape": "4x4", "numel": 16, "l2_norm": 3.5, "modified": false},
                {"name": "b.weight", "shape": "4x4", "numel": 16, "l2_norm": 2.25, "modified": true}
            ],
            "matched_count": 2
        });
        let s = dumps_toon(&v);
        let back = loads_toon(&s);
        assert_eq!(back["matched_count"], json!(2));
        assert_eq!(back["tensors"][0]["name"], json!("a.weight"));
        assert_eq!(back["tensors"][1]["modified"], json!(true));
    }

    #[test]
    fn roundtrip_nested_list() {
        let v = json!({
            "outlier_neurons": [
                {"layer": "layer_2", "tensor_name": "w", "row_indices": [5, 11, 19], "max_ratio_to_median": 16.0, "median_row_norm": 0.5}
            ]
        });
        let s = dumps_toon(&v);
        let back = loads_toon(&s);
        assert_eq!(back["outlier_neurons"][0]["row_indices"], json!([5, 11, 19]));
        assert_eq!(back["outlier_neurons"][0]["layer"], json!("layer_2"));
    }

    #[test]
    fn numeric_strings_and_float_precision() {
        // Regression: numeric-looking strings must stay strings, and floats must
        // round-trip without %.6g precision loss (interop with Python toon_format).
        let v = json!({"t": [{"shape": "576", "l2_norm": 181.0093, "big": 716.5232}]});
        let s = dumps_toon(&v);
        assert!(s.contains("\"576\""), "numeric string must be quoted: {s}");
        assert!(s.contains("181.0093"), "float precision lost: {s}");
        let back = loads_toon(&s);
        assert_eq!(back["t"][0]["shape"], json!("576"));
        assert_eq!(back["t"][0]["l2_norm"], json!(181.0093));
        assert_eq!(back["t"][0]["big"], json!(716.5232));
    }

    #[test]
    fn token_savings_positive_on_tables() {
        let rows: Vec<Value> = (0..20)
            .map(|i| json!({"name": format!("model.layers.{i}.mlp.down_proj.weight"), "numel": 110592, "l2_norm": 12.34}))
            .collect();
        let rep = compare_toon_vs_json(&json!({"tensors": rows}));
        assert!(rep.token_reduction_ratio > 0.2);
    }
}