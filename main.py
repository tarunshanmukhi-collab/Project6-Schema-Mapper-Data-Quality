import streamlit as st
import pandas as pd
import json
import os
import re
from rapidfuzz import process, fuzz
from datetime import datetime, timezone
from fpdf import FPDF
import io

# Configure Streamlit as the first Streamlit call (best practice)
st.set_page_config(page_title="Schema Mapper & Data Quality Fixer", layout="wide")

# ---------- Schema Management ----------
SCHEMA_DIR = "schemas"
os.makedirs(SCHEMA_DIR, exist_ok=True)

def list_schema_files():
    return [f for f in os.listdir(SCHEMA_DIR) if f.endswith('.csv')]

def load_schema(schema_file):
    df = pd.read_csv(os.path.join(SCHEMA_DIR, schema_file))
    return list(df["canonical_name"])

def save_schema(schema_file, fields):
    df = pd.DataFrame({"canonical_name": fields})
    df.to_csv(os.path.join(SCHEMA_DIR, schema_file), index=False)

def delete_schema(schema_file):
    os.remove(os.path.join(SCHEMA_DIR, schema_file))


# ---------- Config / Persistence ----------
PROMOTED_FILE = "promoted_rules.json"

# --- Schema Selection and Editing ---
st.sidebar.header("Schema Management")
schema_files = list_schema_files()
if not schema_files:
    # If no schemas, create a default one from Project6StdFormat.csv if exists
    if os.path.exists("Project6StdFormat.csv"):
        df = pd.read_csv("Project6StdFormat.csv")
        df.to_csv(os.path.join(SCHEMA_DIR, "Project6StdFormat.csv"), index=False)
        schema_files = ["Project6StdFormat.csv"]
    else:
        st.sidebar.warning("No schema files found. Please upload or create one.")

selected_schema = st.sidebar.selectbox("Select Canonical Schema", schema_files, index=0 if schema_files else None, key="schema_select")

# if st.sidebar.button("Upload New Schema CSV"):
#     uploaded_schema = st.sidebar.file_uploader("Upload Schema CSV", type=["csv"], key="schema_upload")
#     if uploaded_schema is not None:
#         with open(os.path.join(SCHEMA_DIR, uploaded_schema.name), "wb") as f:
#             f.write(uploaded_schema.read())
#         st.sidebar.success(f"Uploaded {uploaded_schema.name}")
#         st.rerun()

if selected_schema:
    CANONICAL_SCHEMA_FILE = os.path.join(SCHEMA_DIR, selected_schema)
    canonical_df = pd.read_csv(CANONICAL_SCHEMA_FILE)
    CANONICAL_SCHEMA = list(canonical_df["canonical_name"])
else:
    CANONICAL_SCHEMA = []

# --- Schema Editor UI ---
st.sidebar.markdown("---")
st.sidebar.subheader("Edit Canonical Schema")
if selected_schema:
    schema_fields = CANONICAL_SCHEMA.copy()
    st.sidebar.write("Current fields:")
    for i, field in enumerate(schema_fields):
        col1, col2 = st.sidebar.columns([4,1])
        col1.write(field)
        if col2.button("❌", key=f"del_{field}"):
            schema_fields.pop(i)
            save_schema(selected_schema, schema_fields)
            st.rerun()
    new_field = st.sidebar.text_input("Add new field", key="add_field")
    if new_field:
        if new_field not in schema_fields:
            schema_fields.append(new_field)
            save_schema(selected_schema, schema_fields)
            st.rerun()
        else:
            st.sidebar.warning("Field already exists.")
    if st.sidebar.button("Delete Schema File"):
        delete_schema(selected_schema)
        st.sidebar.success(f"Deleted {selected_schema}")
        st.rerun()



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

# # --- Custom Cleaner UI ---
# st.sidebar.markdown("---")
# st.sidebar.subheader("Custom Cleaning Rules")
# if CANONICAL_SCHEMA:
#     custom_col = st.sidebar.selectbox("Column for custom cleaner", CANONICAL_SCHEMA, key="custom_clean_col")
#     promoted = load_promoted()  # reload in case of changes
#     custom_cleaners = promoted.get("custom_cleaners", {})
#     existing_code = custom_cleaners.get(custom_col, "def custom_clean(val):\n    # val is the cell value\n    return val")
#     custom_code = st.sidebar.text_area("Python function for cleaning (edit as needed)", value=existing_code, height=120, key="custom_clean_code")
#     if st.sidebar.button("Save Custom Cleaner"):
#         # Save code
#         promoted.setdefault("custom_cleaners", {})[custom_col] = custom_code
#         save_promoted(promoted)
#         st.sidebar.success(f"Custom cleaner saved for {custom_col}")
#     if custom_col in custom_cleaners:
#         if st.sidebar.button("Remove Custom Cleaner"):
#             promoted["custom_cleaners"].pop(custom_col, None)
#             save_promoted(promoted)
#             st.sidebar.success(f"Custom cleaner removed for {custom_col}")


# Ensure promoted is loaded before main app logic
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
    promoted = load_promoted()
    custom_cleaners = promoted.get("custom_cleaners", {})
    for col in df.columns:
        cleaner = None
        # Priority: custom cleaner > promoted rule > default cleaner
        if col in custom_cleaners:
            # Dynamically define the function
            local_vars = {}
            try:
                exec(custom_cleaners[col], {}, local_vars)
                custom_fn = local_vars.get("custom_clean")
                if custom_fn:
                    df[col] = df[col].apply(lambda x, fn=custom_fn: fn(x))
                    continue
            except Exception as e:
                st.warning(f"Error in custom cleaner for {col}: {e}")
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
uploaded = st.file_uploader("Upload a CSV", type=["csv", "txt"], accept_multiple_files=False)

if uploaded:
    if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded.name:
        st.session_state.user_mapping = None
        st.session_state.last_uploaded_name = uploaded.name
    st.subheader("Preview of uploaded file")
    df_raw = pd.read_csv(uploaded)
    st.dataframe(df_raw)

    st.markdown("---")
    st.header("Schema Mapping Suggestions")

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

    st.markdown("---")
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

    if "last_issues" in st.session_state:
        st.markdown("---")
        st.header("Targeted Fix Queue (Leftover Issues)")
        issues = st.session_state.last_issues
        # --- Download summary report (CSV) ---
        if issues:
            import io
            issues_df = pd.DataFrame(issues)
            csv_issues = issues_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Issues Report (CSV)", data=csv_issues, file_name="issues_report.csv", mime="text/csv")
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
                st.rerun()

    st.markdown("---")
    st.header("Finalize & Export")
    if "last_cleaned" in st.session_state:
        st.write("Preview cleaned data (top 20 rows):")
        st.dataframe(st.session_state.last_cleaned.head(20))
        csv = st.session_state.last_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned CSV", data=csv, file_name="cleaned_output.csv", mime="text/csv")

    # --- PDF Download Button ---


        def df_to_pdf(df, title="Cleaned Data"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=title, ln=True, align="C")
            # Guard for empty data to avoid layout issues
            if df is None or df.empty or len(df.columns) == 0:
                pdf.ln(10)
                pdf.cell(200, 10, txt="No data to display.", ln=True, align="C")
                pdf_bytes = pdf.output(dest='S').encode('latin1', errors='replace')
                return io.BytesIO(pdf_bytes)
            col_width = pdf.w / (len(df.columns) + 1)
            row_height = pdf.font_size * 1.5
            # Header
            for col in df.columns:
                pdf.cell(col_width, row_height, str(col), border=1)
            pdf.ln(row_height)
            # Rows (limit to 20 rows for PDF)
            for i, row in df.head(20).iterrows():
                for item in row:
                    pdf.cell(col_width, row_height, str(item), border=1)
                pdf.ln(row_height)
            # fpdf 1.x requires latin-1; replace unsupported characters gracefully
            pdf_bytes = pdf.output(dest='S').encode('latin1', errors='replace')
            return io.BytesIO(pdf_bytes)

        pdf_buffer = df_to_pdf(st.session_state.last_cleaned)
        st.download_button("Download cleaned PDF", data=pdf_buffer, file_name="cleaned_output.pdf", mime="application/pdf")

        st.markdown("**Promote column mappings for this partner**")
        if st.button("Promote current mappings"):
            promoted.setdefault("column_mappings", {})
            for src, tgt in st.session_state.last_mapping.items():
                promoted["column_mappings"][src] = tgt
            save_promoted(promoted)
            st.success("Current mappings promoted. Future uploads with same headers will auto-map.")
            promoted = load_promoted()
            st.rerun()

else:
    st.info("Upload a CSV file to get started. (Only CSVs are supported.)")
    st.markdown("Example canonical schema used in this demo:")
    st.write(CANONICAL_SCHEMA)

st.markdown("---")
st.caption(f"App run at {datetime.now(timezone.utc).isoformat()}Z. Mappings & cleaning rules persisted to `{PROMOTED_FILE}`.")