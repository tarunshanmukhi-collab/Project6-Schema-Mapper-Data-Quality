import streamlit as st
import pandas as pd
import json
import os
import re
from rapidfuzz import process, fuzz
from datetime import datetime, timezone

# ---------- Config / Persistence ----------
PROMOTED_FILE = "promoted_rules.json"

# --- Simple state bootstrap ---
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None 
if "mapping" not in st.session_state:
    st.session_state.mapping = {}           # {raw_col -> canonical_col}
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "dq_report" not in st.session_state:
    st.session_state.dq_report = {}         # metrics & leftover issues
if "rules_pack" not in st.session_state:
    st.session_state.rules_pack = [] 
if "promoted" not in st.session_state or st.session_state.promoted is None:
    st.session_state.promoted = {}

# Load canonical schema from CSV
CANONICAL_SCHEMA_FILE = "Project6StdFormat.csv"
if not os.path.exists(CANONICAL_SCHEMA_FILE):
    st.error(f"Schema file `{CANONICAL_SCHEMA_FILE}` not found. Please provide Project6StdFormat.csv")
    st.stop()

canonical_df = pd.read_csv(CANONICAL_SCHEMA_FILE)
CANONICAL_SCHEMA = list(canonical_df["canonical_name"])

# ensure persistence file exists
if not os.path.exists(PROMOTED_FILE):
    with open(PROMOTED_FILE, "w") as f:
        json.dump({"column_mappings": {}, "cleaning_rules": {}}, f, indent=2)

def load_promoted():
    with open(PROMOTED_FILE, "r") as f:
        return json.load(f)

def save_promoted(data):
    with open(PROMOTED_FILE, "w") as f:
        json.dump(data, f, indent=2)

promoted = load_promoted()

# ---------- Utility functions ----------

def normalize_header(h: str) -> str:
    """Normalize header for matching: lower, strip punctuation."""
    if pd.isna(h):
        return ""
    h = str(h).lower().strip()
    h = re.sub(r"[_\-\s]+", " ", h)
    h = re.sub(r"[^\w\s]", "", h)
    return h

def suggest_mapping(headers, canonical=CANONICAL_SCHEMA, promoted_map=None, top_n=3):
    promoted_map = promoted_map or {}
    suggestions = []
    canon_norm = {c: normalize_header(c) for c in canonical}
    use_transformers = st.session_state.get("use_transformers", False)
    if use_transformers:
        try:
            from sentence_transformers import SentenceTransformer, util
            model = st.session_state.get("st_model")
            if model is None:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                st.session_state["st_model"] = model
            canon_embeds = model.encode(list(canon_norm.values()), convert_to_tensor=True)
        except Exception as e:
            st.warning(f"SentenceTransformers not available: {e}. Falling back to rapidfuzz.")
            use_transformers = False
    for h in headers:
        hnorm = normalize_header(h)
        # check promoted (exact or normalized)
        if h in promoted_map.get("column_mappings", {}):
            suggestions.append((h, promoted_map["column_mappings"][h], 100.0, "promoted"))
            continue
        candidates = []
        if use_transformers:
            try:
                h_embed = model.encode(hnorm, convert_to_tensor=True)
                sims = util.pytorch_cos_sim(h_embed, canon_embeds)[0].cpu().numpy()
                for i, c in enumerate(canonical):
                    candidates.append((c, float(sims[i]*100)))
            except Exception as e:
                st.warning(f"SentenceTransformers error: {e}. Using rapidfuzz fallback.")
                use_transformers = False
        if not use_transformers:
            for c, cnorm in canon_norm.items():
                score = fuzz.token_sort_ratio(hnorm, cnorm)  # 0-100
                candidates.append((c, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[0]
        suggestions.append((h, top[0], float(top[1]), "transformers" if use_transformers else "fuzzy"))
    return suggestions

# ---------- Deterministic cleaners for canonical fields ----------

def clean_order_id(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    return s if s else None

def clean_order_date(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    # Try to parse date in various formats
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    # Try pandas parse
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

def clean_customer_id(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    return s if s else None

def clean_customer_name(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s.title()

def clean_email(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    return s

def clean_phone(v):
    if pd.isna(v):
        return None
    s = re.sub(r"[^\d+]", "", str(v))
    return s if len(s) >= 7 else None

def clean_billing_address(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def clean_shipping_address(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def clean_city(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s.title()

def clean_state(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s.title()

def clean_postal_code(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    s = re.sub(r"[^\d]", "", s)
    return s if s else None

def clean_country(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    # Normalize common variants
    if s.lower() in ["usa", "us", "united states", "united states of america"]:
        return "United States"
    return s.title()

def clean_product_sku(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    return s

def clean_product_name(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s.title()

def clean_category(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s.title()

def clean_subcategory(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s.title()

def clean_quantity(v):
    if pd.isna(v):
        return None
    try:
        val = int(float(str(v).replace(",", "").strip()))
        return val if val >= 0 else None
    except Exception:
        return None

def clean_unit_price(v):
    if pd.isna(v):
        return None
    s = str(v).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except Exception:
        return None

def clean_currency(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    return s

def clean_discount_pct(v):
    if pd.isna(v):
        return None
    s = str(v).replace("%", "").strip()
    try:
        val = float(s)
        if val > 1:  # If percent, convert to fraction
            val = val / 100.0
        return round(val, 4)
    except Exception:
        return None

def clean_tax_pct(v):
    if pd.isna(v):
        return None
    s = str(v).replace("%", "").strip()
    try:
        val = float(s)
        if val > 1:
            val = val / 100.0
        return round(val, 4)
    except Exception:
        return None

def clean_shipping_fee(v):
    if pd.isna(v):
        return None
    s = str(v).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except Exception:
        return None

def clean_total_amount(v):
    if pd.isna(v):
        return None
    s = str(v).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except Exception:
        return None

def clean_tax_id(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    s = re.sub(r"[^\dA-Za-z]", "", s)
    return s if s else None

DEFAULT_CLEANERS = {
    "order_id": clean_order_id,
    "order_date": clean_order_date,
    "customer_id": clean_customer_id,
    "customer_name": clean_customer_name,
    "email": clean_email,
    "phone": clean_phone,
    "billing_address": clean_billing_address,
    "shipping_address": clean_shipping_address,
    "city": clean_city,
    "state": clean_state,
    "postal_code": clean_postal_code,
    "country": clean_country,
    "product_sku": clean_product_sku,
    "product_name": clean_product_name,
    "category": clean_category,
    "subcategory": clean_subcategory,
    "quantity": clean_quantity,
    "unit_price": clean_unit_price,
    "currency": clean_currency,
    "discount_pct": clean_discount_pct,
    "tax_pct": clean_tax_pct,
    "shipping_fee": clean_shipping_fee,
    "total_amount": clean_total_amount,
    "tax_id": clean_tax_id,
}

def apply_cleaners(df, mapping, cleaning_rules):
    df = df.copy()
    # Only rename mapped columns, leave others as-is
    rename_map = {src: tgt for src, tgt in mapping.items() if tgt}
    df = df.rename(columns=rename_map)
    # Clean only canonical columns
    for col in df.columns:
        cleaner = None
        if col in cleaning_rules:
            rule = cleaning_rules[col]
            cleaner = DEFAULT_CLEANERS.get(rule, None)
        elif col in DEFAULT_CLEANERS:
            cleaner = DEFAULT_CLEANERS.get(col, None)
        if cleaner:
            df[col] = df[col].apply(lambda x, fn=cleaner: fn(x))
        # else: leave unmapped/extra columns untouched
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x, fn=str.strip: fn(x) if isinstance(x, str) else x)
    return df

def validate_df(df):
    issues = []
    for idx, row in df.iterrows():
        # Example: required fields
        for col in ["order_id", "order_date", "customer_id", "customer_name", "email", "product_sku", "quantity", "unit_price", "total_amount"]:
            if col in df.columns:
                v = row.get(col)
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    issues.append({
                        "row": int(idx),
                        "column": col,
                        "issue": "missing",
                        "message": f"{col} missing or blank",
                        "value": v,
                        "suggestions": ["leave_blank", "enter_manually"]
                    })
        # Email format
        if "email" in df.columns:
            e = row.get("email")
            if e and isinstance(e, str) and not re.match(r"[^@]+@[^@]+\.[^@]+", e):
                issues.append({
                    "row": int(idx),
                    "column": "email",
                    "issue": "invalid_format",
                    "message": "Email doesn't look valid",
                    "value": e,
                    "suggestions": ["edit", "leave_blank"]
                })
        # Postal code should be digits
        if "postal_code" in df.columns:
            pc = row.get("postal_code")
            if pc and (not str(pc).isdigit() or len(str(pc)) < 4):
                issues.append({
                    "row": int(idx),
                    "column": "postal_code",
                    "issue": "invalid_format",
                    "message": "Postal code should be digits only and at least 4 digits",
                    "value": pc,
                    "suggestions": ["edit", "leave_blank"]
                })
        # Percent fields
        for col in ["discount_pct", "tax_pct"]:
            if col in df.columns:
                v = row.get(col)
                if v is not None and (v < 0 or v > 1):
                    issues.append({
                        "row": int(idx),
                        "column": col,
                        "issue": "invalid_range",
                        "message": f"{col} should be between 0 and 1",
                        "value": v,
                        "suggestions": ["edit", "leave_blank"]
                    })
        # Quantity should be non-negative
        if "quantity" in df.columns:
            q = row.get("quantity")
            if q is not None and q < 0:
                issues.append({
                    "row": int(idx),
                    "column": "quantity",
                    "issue": "invalid_value",
                    "message": "Quantity cannot be negative",
                    "value": q,
                    "suggestions": ["edit", "leave_blank"]
                })
    return issues

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Schema Mapper & Data Quality Fixer", layout="wide")
st.title("Schema Mapper & Data Quality Fixer (Map → Clean → Targeted Repair)")

st.sidebar.header("Quick Controls")
st.sidebar.markdown("""
- Upload a CSV
- Confirm/override suggested mappings
- Run clean & validate
- Apply/approve fixes and promote rules
""")
# Add toggle for similarity method
st.sidebar.markdown("---")
st.sidebar.write("**Similarity Method**")
sim_method = st.sidebar.radio("Choose mapping similarity method:", ["RapidFuzz", "SentenceTransformers"], index=0, key="sim_method_toggle")
st.session_state["use_transformers"] = (sim_method == "SentenceTransformers")

def page_upload():
    uploaded = st.file_uploader("Upload a CSV", type=["csv", "txt"], accept_multiple_files=False)
    if uploaded:
        if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded.name:
            st.session_state.user_mapping = None
            st.session_state.last_uploaded_name = uploaded.name
        st.subheader("Preview of uploaded file")
        df_raw = pd.read_csv(uploaded)
        st.session_state.raw_df = df_raw
        st.dataframe(df_raw)
        if st.button("Suggest mapping", icon=":material/lightbulb:"):
            st.rerun()
    else:
        st.info("Upload a CSV file to get started. (Only CSVs are supported.)")
        st.markdown("Example canonical schema used in this demo:")
        st.write(CANONICAL_SCHEMA)

def page_mapping_v0():
    st.markdown("---")
    st.header("Schema Mapping Suggestions")
    df_raw = st.session_state.raw_df
    headers = list(df_raw.columns)
    suggestions = suggest_mapping(headers, CANONICAL_SCHEMA, promoted_map=promoted)

    st.write("Suggested mappings (you can override):")
    mapping_cols = {}
    cols = st.columns((3, 3, 2, 2))
    cols[0].markdown("**Source Column**")
    cols[1].markdown("**Suggested Canonical Field**")
    cols[2].markdown("**Confidence**")
    cols[3].markdown("**Override**")

    if "user_mapping" not in st.session_state or st.session_state.user_mapping is None:
        st.session_state.user_mapping = {s[0]: s[1] for s in suggestions}

    for src, suggested, conf, how in suggestions:
        c0, c1, c2, c3 = st.columns((3, 3, 2, 2))
        c0.write(f"**{src}**")
        options = ["(ignore)"] + CANONICAL_SCHEMA
        sel = c1.selectbox(f"map_{src}", options, index=(options.index(st.session_state.user_mapping.get(src)) if st.session_state.user_mapping.get(src) in options else 0), key=f"map_{src}_key")
        c2.write(f"{conf:.0f}% {'(promoted)' if how=='promoted' else ''}")
        if sel == "(ignore)":
            st.session_state.user_mapping[src] = None
        else:
            st.session_state.user_mapping[src] = sel

    targets = [v for v in st.session_state.user_mapping.values() if v]
    dup_targets = {t for t in targets if targets.count(t) > 1}
    if dup_targets:
        st.warning(f"Multiple source columns mapped to same canonical target: {', '.join(dup_targets)}. This may cause overwriting after rename. Please resolve before continuing.")
        st.stop()
    else:
        st.success("No duplicate target mappings detected.")
        if st.button("Save mapping", type="primary", icon=":material/save:"):
            st.session_state.mapping = True
    # st.session_state.raw_df = df_raw

def page_mapping_v1():
    import pandas as pd
    st.markdown("---")
    st.header("Schema Mapping Suggestions")

    df_raw = st.session_state.raw_df
    headers = list(df_raw.columns)

    # suggestions: iterable of (src, suggested, conf, how)
    suggestions = suggest_mapping(headers, CANONICAL_SCHEMA, promoted_map=promoted)

    # ---- Build a work DataFrame for the editor ----
    rows = []
    for src, suggested, conf, how in suggestions:
        # Normalize confidence: handle values in [0,1] vs [0,100]
        if conf is None:
            conf_norm = 0
        else:
            conf_norm = conf * 100 if conf <= 1.0 else conf
        label = f"{conf_norm:.0f}%"
        if how == "promoted":
            label += " (promoted)"
        rows.append({
            "source_col": src,
            "suggested": suggested if suggested else "(ignore)",
            "confidence": float(conf_norm),
            "confidence_label": label
        })
    work = pd.DataFrame(rows)

    # Initialize or reuse previous mapping
    if "user_mapping" not in st.session_state or st.session_state.user_mapping is None:
        st.session_state.user_mapping = {
            r["source_col"]: (None if r["suggested"] == "(ignore)" else r["suggested"])
            for _, r in work.iterrows()
        }

    # Canonical column shown/edited in the grid (defaults to prior selection or suggested)
    def _init_choice(src, suggested):
        saved = st.session_state.user_mapping.get(src)
        return saved if saved is not None else "(ignore)" if suggested == "(ignore)" else suggested

    work["canonical"] = [
        _init_choice(r.source_col, r.suggested) for r in work.itertuples(index=False)
    ]

    # Options for the selectbox column
    options = ["(ignore)"] + list(CANONICAL_SCHEMA)

    edited = st.data_editor(
        work[["source_col", "suggested", "canonical", "confidence"]],
        hide_index=True,
        num_rows="fixed",
        width='stretch',
        height=400,
        column_config={
            "source_col": st.column_config.TextColumn("Source Column", disabled=True),
            "suggested": st.column_config.TextColumn("Suggested", disabled=True),
            "canonical": st.column_config.SelectboxColumn(
                "Chosen Canonical",
                options=options
            ),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=100, format="%.0f%%"
            ),
        }
    )

    # Update session mapping from edits
    new_map = {}
    for _, r in edited.iterrows():
        choice = r["canonical"]
        new_map[r["source_col"]] = None if choice == "(ignore)" else choice
    st.session_state.user_mapping = new_map

    # ---- Duplicate target check ----
    targets = [v for v in new_map.values() if v]
    dup_targets = {t for t in targets if targets.count(t) > 1}
    if dup_targets:
        st.warning(
            "Multiple source columns mapped to the same canonical target: "
            + ", ".join(sorted(dup_targets))
            + ". This may cause overwriting after rename. Please resolve before continuing."
        )
        st.stop()
    else:
        st.success("No duplicate target mappings detected.")
        if st.button("Save mapping", type="primary", icon=":material/save:"):
            # Gate later pages with this flag (as in your existing nav logic)
            st.session_state.mapping = True


def page_clean_validate():
    df_raw = st.session_state.raw_df
    st.header("One-Click Clean & Validate")
    if st.button("Run Clean & Validate"):
        mapping = {src: tgt for src, tgt in st.session_state.user_mapping.items() if tgt}
        st.session_state.last_mapping = mapping
        st.subheader("Before (first 10 rows)")
        st.dataframe(df_raw)

        cleaned = apply_cleaners(df_raw, mapping, promoted.get("cleaning_rules", {}))
        st.subheader("After Clean (first 10 rows)")

        # Highlight changed values (text color only)
        def highlight_text(cleaned_val, raw_val):
            if pd.isna(raw_val) and pd.isna(cleaned_val):
                return ""
            if raw_val != cleaned_val:
                return "color: #d9534f"  # Bootstrap red
            return ""

        raw_display = df_raw.copy()
        cleaned_display = cleaned.copy()

        def style_func(col):
            # Compare each cell in cleaned with raw
            return [
                highlight_text(cleaned_display.at[idx, col.name], 
                               raw_display.at[idx, col.name] if col.name in raw_display.columns else None)
                for idx in cleaned_display.index
            ]

        styled = cleaned_display.style.apply(style_func, axis=0)
        st.dataframe(styled)

        issues = validate_df(cleaned)
        st.session_state.last_cleaned = cleaned
        st.session_state.last_issues = issues

        st.success(f"Cleaning complete. Found {len(issues)} issue(s). Scroll down for targeted fixes.")
        st.markdown("**Before/After metrics**")
        before_counts = df_raw.count().to_frame("before_count")
        after_counts = cleaned.count().to_frame("after_count")
        metrics = before_counts.join(after_counts, how="outer").fillna(0).astype(int)
        st.table(metrics)
        st.session_state.cleaned_df = cleaned

def page_mapping():
    import pandas as pd

    st.markdown("---")
    st.header("Schema Mapping Suggestions")

    df_raw = st.session_state.raw_df
    headers = list(df_raw.columns)

    # suggestions: iterable of (src, suggested, conf, how)
    suggestions = suggest_mapping(headers, CANONICAL_SCHEMA, promoted_map=promoted)

    # ---- Build a work DataFrame for the editor ----
    rows = []
    for src, suggested, conf, how in suggestions:
        # Normalize confidence: handle values in [0,1] vs [0,100]
        if conf is None:
            conf_norm = 0
        else:
            conf_norm = conf * 100 if conf <= 1.0 else conf
        rows.append({
            "source_col": src,
            "suggested": suggested if suggested else "(ignore)",
            "confidence": float(conf_norm),
        })
    work = pd.DataFrame(rows)

    # Initialize or reuse previous mapping
    if "user_mapping" not in st.session_state or st.session_state.user_mapping is None:
        st.session_state.user_mapping = {
            r["source_col"]: (None if r["suggested"] == "(ignore)" else r["suggested"])
            for _, r in work.iterrows()
        }

    # Canonical column shown/edited in the grid (defaults to prior selection or suggested)
    def _init_choice(src, suggested):
        saved = st.session_state.user_mapping.get(src)
        return saved if saved is not None else "(ignore)" if suggested == "(ignore)" else suggested

    work["canonical"] = [_init_choice(r.source_col, r.suggested) for r in work.itertuples(index=False)]

    # ---- Compute duplicates based on current choices (from last run) ----
    chosen_now = [v for v in work["canonical"].tolist() if v and v != "(ignore)"]
    dup_targets = {t for t in chosen_now if chosen_now.count(t) > 1}

    # Single editor with an in-table duplicate flag
    work["duplicate"] = work["canonical"].apply(lambda x: "⚠️ Yes" if x in dup_targets else "")

    options = ["(ignore)"] + list(CANONICAL_SCHEMA)

    edited = st.data_editor(
        work[["source_col", "suggested", "canonical", "confidence", "duplicate"]],
        hide_index=True,
        num_rows="fixed",
        width='stretch',
        height=500,
        column_config={
            "source_col": st.column_config.TextColumn("Source Column", disabled=True),
            "suggested": st.column_config.TextColumn("Suggested", disabled=True),
            "canonical": st.column_config.SelectboxColumn("Chosen Canonical", options=options),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.0f%%"),
            "duplicate": st.column_config.TextColumn("Duplicate", disabled=True),
        },
        key="mapping_editor"  # single editor, single key
    )

    # Update session mapping from this edit (so next rerun shows updated duplicate flags)
    new_map = {}
    for _, r in edited.iterrows():
        choice = r["canonical"]
        new_map[r["source_col"]] = None if choice == "(ignore)" else choice
    st.session_state.user_mapping = new_map

    # ---- Duplicate target check + Save gating ----
    targets = [v for v in new_map.values() if v]
    dup_targets = {t for t in targets if targets.count(t) > 1}

    if dup_targets:
        st.warning(
            "Multiple source columns mapped to the same canonical target: "
            + ", ".join(sorted(dup_targets))
            + ". This may cause overwriting after rename. Please resolve before continuing."
        )
        st.stop()
    else:
        st.success("No duplicate target mappings detected.")
        if st.button("Save mapping", type="primary", icon=":material/save:"):
            st.session_state.mapping = True


def page_targeted_fix():
    if "last_issues" in st.session_state:
        st.markdown("---")
        st.header("Targeted Fix Queue (Leftover Issues)")
        issues = st.session_state.last_issues
        if not issues:
            st.info("No remaining issues found. You're good to go.")
        else:
            fix_results = []
            for i, iss in enumerate(issues):
                st.markdown(f"**Issue {i+1}: Row {iss['row']} — {iss['column']}**")
                st.write(iss["message"])
                st.write("Current value:", iss["value"])
                action = st.selectbox(f"action_{i}", options=iss["suggestions"] + ["copy_from_other_column"], index=0, key=f"action_{i}_key")
                manual_val = None
                if action == "enter_manually" or action == "edit":
                    manual_val = st.text_input(f"manual_value_{i}", key=f"manual_val_{i}")
                elif action == "copy_from_other_column":
                    other_cols = list(st.session_state.last_cleaned.columns)
                    sel_col = st.selectbox(f"copy_col_{i}", options=other_cols, key=f"copy_col_{i}_key")
                    manual_val = sel_col
                fix_results.append({
                    "issue_index": i,
                    "row": iss["row"],
                    "column": iss["column"],
                    "action": action,
                    "manual_val": manual_val
                })
            if st.button("Apply Selected Fixes"):
                cleaned = st.session_state.last_cleaned.copy()
                for fr in fix_results:
                    r = fr["row"]
                    c = fr["column"]
                    act = fr["action"]
                    mv = fr["manual_val"]
                    if act in ("leave_blank",):
                        cleaned.at[r, c] = None
                    elif act in ("enter_manually", "edit"):
                        cleaned.at[r, c] = mv if mv != "" else None
                    elif act == "copy_from_other_column" and mv in cleaned.columns:
                        cleaned.at[r, c] = cleaned.at[r, mv]
                new_issues = validate_df(cleaned)
                st.session_state.last_cleaned = cleaned
                st.session_state.last_issues = new_issues
                st.success("Applied fixes. Validation re-run.")
                st.rerun()

            st.markdown("**Promote cleaning rules (so they auto-run on future files)**")
            promote_cols = []
            for col in st.session_state.last_cleaned.columns:
                if col in DEFAULT_CLEANERS:
                    checked = st.checkbox(f"Promote default cleaner for `{col}` ({DEFAULT_CLEANERS[col].__name__})", key=f"promote_{col}")
                    if checked:
                        promote_cols.append((col, DEFAULT_CLEANERS[col].__name__))
            if st.button("Save Promoted Cleaners"):
                for col, rule_name in promote_cols:
                    promoted.setdefault("cleaning_rules", {})[col] = rule_name
                save_promoted(promoted)
                st.success("Promoted cleaning rules saved.")
                promoted = load_promoted()
                st.session_state.promoted = promoted
                st.rerun()

def page_export():
    st.header("Finalize & Export")
    if "last_cleaned" in st.session_state:
        st.write("Preview cleaned data (top 20 rows):")
        st.dataframe(st.session_state.last_cleaned.head(20))
        csv = st.session_state.last_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned CSV", data=csv, file_name="cleaned_output.csv", mime="text/csv")
        st.markdown("**Promote column mappings for this partner**")
        if st.button("Promote current mappings"):
            promoted = st.session_state.promoted
            promoted.setdefault("column_mappings", {})
            for src, tgt in st.session_state.last_mapping.items():
                promoted["column_mappings"][src] = tgt
            save_promoted(promoted)
            st.success("Current mappings promoted. Future uploads with same headers will auto-map.")
            promoted = load_promoted()
            st.rerun()




# # ---------- Build dynamic nav: gate later pages until earlier steps complete ----------
upload = st.Page(page_upload, title="Upload", icon=":material/upload:")
mapping = st.Page(page_mapping, title="Mapping", icon=":material/sync_alt:")
clean   = st.Page(page_clean_validate, title="Clean & Validate", icon=":material/sweep:")
fixes   = st.Page(page_targeted_fix, title="Targeted Fix", icon=":material/build:")
export  = st.Page(page_export, title="Report & Export", icon=":material/download:")

pages = [upload]
if st.session_state.raw_df is not None:
    pages.append(mapping)
    pages.append(clean)
    pages.append(fixes)
    pages.append(export)

pg = st.navigation(pages, position="top")   # also accepts "sidebar" or "hidden"
st.markdown("---")
st.caption(f"App run at {datetime.now(timezone.utc).isoformat()}Z. Mappings & cleaning rules persisted to `{PROMOTED_FILE}`.")
pg.run()