import streamlit as st
import pandas as pd
import json
import os
import re
from rapidfuzz import fuzz
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import openai
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer, util
from numpy import dot
from numpy.linalg import norm

load_dotenv()

# ---------- Config / Persistence ----------
PROMOTED_FILE = "promoted_rules.json"
CANONICAL_SCHEMA_FILE = "Project6StdFormat.csv"

# Ensure persistence file exists with correct structure
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

# Check if canonical schema file exists
if not os.path.exists(CANONICAL_SCHEMA_FILE):
    st.error(f"Schema file `{CANONICAL_SCHEMA_FILE}` not found. Please provide <Project6StdFormat.csv> with canonical_name and description columns.")
    st.stop()

canonical_df = pd.read_csv(CANONICAL_SCHEMA_FILE)
CANONICAL_SCHEMA = list(canonical_df["canonical_name"])
CANONICAL_DESCRIPTIONS = dict(zip(
    canonical_df["canonical_name"],
    canonical_df["description"] if "description" in canonical_df.columns else [""]*len(canonical_df)
))

# ---------- Utility functions ----------

def normalize_header(h: str) -> str:
    """Normalize header for matching: lower, strip punctuation."""
    if pd.isna(h):
        return ""
    h = str(h).lower().strip()
    # h = re.sub(r"[_\-\s]+", " ", h)
    # h = re.sub(r"[^\w\s]", "", h)
    return h

def suggest_mapping(headers, canonical=CANONICAL_SCHEMA, promoted_map=None, top_n=3):
    promoted_map = promoted_map or {}
    suggestions = []
    canon_norm = {c: normalize_header(c) for c in canonical}
    sim_method = st.session_state.get("sim_method", "RapidFuzz")
    use_transformers = (sim_method == "SentenceTransformers")
    use_hf_tokenizer = (sim_method == "HF AutoTokenizer")
    use_openai_embeddings = (sim_method == "OpenAI Embeddings")
    use_hf_embeddings = (sim_method == "HuggingFaceEmbeddings")

    # HuggingFaceEmbeddings
    if use_hf_embeddings:
        try:
            if "hf_embedder" not in st.session_state:
                st.session_state["hf_embedder"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embedder = st.session_state["hf_embedder"]
            if "hf_embed_canon" not in st.session_state:
                canon_texts = list(canon_norm.values())
                # canon_texts = [f"{c} - {CANONICAL_DESCRIPTIONS.get(c, '')}".strip() for c in canonical]
                st.session_state["hf_embed_canon"] = embedder.embed_documents(canon_texts)
            canon_embeds = st.session_state["hf_embed_canon"]
        except Exception as e:
            st.warning(f"HuggingFaceEmbeddings error: {e}. Using rapidfuzz fallback.")
            use_hf_embeddings = False

    # OpenAI Embeddings setup
    if use_openai_embeddings:
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not client.api_key:
                st.warning("OpenAI API key not set. Set OPENAI_API_KEY in environment or Streamlit secrets.")
                use_openai_embeddings = False
            if "openai_canon_embeds" not in st.session_state:
                # canon_texts = list(canon_norm.values())
                canon_texts = [
                    f"{c} - {CANONICAL_DESCRIPTIONS.get(c, '')}".strip()
                    for c in canonical
                ]
                resp = client.embeddings.create(input=canon_texts, model="text-embedding-ada-002")
                st.session_state["openai_canon_embeds"] = [e.embedding for e in resp.data]
        except Exception as e:
            st.warning(f"OpenAI Embeddings error: {e}. Falling back to rapidfuzz.")
            use_openai_embeddings = False

    # SentenceTransformers setup
    if use_transformers:
        try:
            if "st_model" not in st.session_state:
                st.session_state["st_model"] = SentenceTransformer("all-MiniLM-L6-v2")
            model = st.session_state["st_model"]
            if "st_canon_embeds" not in st.session_state:
                canon_texts = list(canon_norm.values())
                st.session_state["st_canon_embeds"] = model.encode(canon_texts, convert_to_tensor=True)
            canon_embeds = st.session_state["st_canon_embeds"]
        except Exception as e:
            st.warning(f"SentenceTransformers error: {e}. Using rapidfuzz fallback.")
            use_transformers = False

    # HF AutoTokenizer setup (dummy example, you should implement get_embedding)
    if use_hf_tokenizer:
        try:
            if "hf_tokenizer" not in st.session_state:
                st.session_state["hf_tokenizer"] = AutoTokenizer.from_pretrained("bert-base-uncased")
                st.session_state["hf_model"] = AutoModel.from_pretrained("bert-base-uncased")
            tokenizer = st.session_state["hf_tokenizer"]
            hf_model = st.session_state["hf_model"]
            def get_embedding(text):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
                with torch.no_grad():
                    outputs = hf_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            if "hf_canon_embeds" not in st.session_state:
                canon_texts = list(canon_norm.values())
                st.session_state["hf_canon_embeds"] = [get_embedding(t) for t in canon_texts]
            canon_embeds = st.session_state["hf_canon_embeds"]
        except Exception as e:
            st.warning(f"HF AutoTokenizer error: {e}. Using rapidfuzz fallback.")
            use_hf_tokenizer = False

    for h in headers:
        hnorm = normalize_header(h)
        # check promoted (exact or normalized)
        if h in promoted_map.get("column_mappings", {}):
            suggestions.append((h, promoted_map["column_mappings"][h], 100.0, "promoted"))
            continue
        candidates = []
        if use_hf_embeddings:
            try:
                h_embed = embedder.embed_query(hnorm)
                canon_embeds = st.session_state["hf_embed_canon"]
                for i, c in enumerate(canonical):
                    sim = dot(h_embed, canon_embeds[i]) / (norm(h_embed) * norm(canon_embeds[i]))
                    candidates.append((c, float(sim*100)))
            except Exception as e:
                st.warning(f"HuggingFaceEmbeddings error: {e}. Using rapidfuzz fallback.")
                use_hf_embeddings = False
        elif use_openai_embeddings:
            try:
                resp = client.embeddings.create(input=[hnorm], model="text-embedding-ada-002")
                h_embed = resp.data[0].embedding
                canon_embeds = st.session_state["openai_canon_embeds"]
                for i, c in enumerate(canonical):
                    sim = dot(h_embed, canon_embeds[i]) / (norm(h_embed) * norm(canon_embeds[i]))
                    candidates.append((c, float(sim*100)))
            except Exception as e:
                st.warning(f"OpenAI Embeddings error: {e}. Using rapidfuzz fallback.")
                use_openai_embeddings = False
        elif use_transformers:
            try:
                h_embed = model.encode(hnorm, convert_to_tensor=True)
                sims = util.pytorch_cos_sim(h_embed, canon_embeds)[0].cpu().numpy()
                for i, c in enumerate(canonical):
                    candidates.append((c, float(sims[i]*100)))
            except Exception as e:
                st.warning(f"SentenceTransformers error: {e}. Using rapidfuzz fallback.")
                use_transformers = False
        elif use_hf_tokenizer:
            try:
                h_embed = get_embedding(hnorm)
                canon_embeds = st.session_state["hf_canon_embeds"]
                for i, c in enumerate(canonical):
                    sim = dot(h_embed, canon_embeds[i]) / (norm(h_embed) * norm(canon_embeds[i]))
                    candidates.append((c, float(sim*100)))
            except Exception as e:
                st.warning(f"HF AutoTokenizer error: {e}. Using rapidfuzz fallback.")
                use_hf_tokenizer = False
        if not use_transformers and not use_hf_tokenizer and not use_openai_embeddings and not use_hf_embeddings:
            for c, cnorm in canon_norm.items():
                score = fuzz.token_sort_ratio(hnorm, cnorm)  # 0-100
                candidates.append((c, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        method_label = (
            "huggingface_embeddings" if use_hf_embeddings else
            "openai_embeddings" if use_openai_embeddings else
            "transformers" if use_transformers else
            ("hf_tokenizer" if use_hf_tokenizer else "fuzzy")
        )
        top = candidates[0]
        suggestions.append((h, top[0], float(top[1]), method_label))
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
sim_method = st.sidebar.radio(
    "Choose mapping similarity method:",
    [
        "RapidFuzz",
        "SentenceTransformers",
        "HF AutoTokenizer",
        "OpenAI Embeddings",
        "HuggingFaceEmbeddings"
    ],
    index=0,
    key="sim_method_toggle"
)
st.session_state["sim_method"] = sim_method
st.session_state["use_transformers"] = (sim_method == "SentenceTransformers")
st.session_state["use_openai_embeddings"] = (sim_method == "OpenAI Embeddings")
st.session_state["use_hf_embeddings"] = (sim_method == "HuggingFaceEmbeddings")

uploaded = st.file_uploader("Upload a CSV", type=["csv", "txt"], accept_multiple_files=False)

def verify_openai_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False

def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if "openai_api_key" in st.session_state:
        api_key = st.session_state["openai_api_key"]
    return api_key

with st.sidebar:
    st.markdown("---")
    st.markdown("**OpenAI API Key**")
    api_key = get_openai_api_key()
    valid = False
    reason = ""
    edit_mode = st.session_state.get("edit_openai_key", False)
    if api_key and not edit_mode:
        valid = verify_openai_key(api_key)
        if valid:
            st.success("OpenAI API key is set and valid.")
            if st.button("✏️ Edit API Key", key="openai_edit_btn"):
                st.session_state["edit_openai_key"] = True
                st.rerun()
        else:
            reason = "The API key provided is invalid or expired."
            st.warning(
                f"To use OpenAI Embeddings, you must provide a valid OpenAI API key.\n\n"
                f"Reason: {reason}\n\n"
                f"Your key will be securely stored in `.env` for future use."
            )
            st.session_state["edit_openai_key"] = True
            st.rerun()
    if not api_key or edit_mode:
        api_key_input = st.text_input("Enter your OpenAI API key", value="", type="password", key="openai_api_key_input")
        if st.button("Verify & Save API Key", key="openai_save_btn"):
            if verify_openai_key(api_key_input):
                st.success("API key is valid and saved.")
                dotenv.set_key(".env", "OPENAI_API_KEY", api_key_input)
                st.session_state["openai_api_key"] = api_key_input
                st.session_state["edit_openai_key"] = False
                st.rerun()
            else:
                st.error("API key is invalid. Please check and try again.")

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
    # Only call suggest_mapping if API key is present for OpenAI Embeddings
    api_key = get_openai_api_key()
    if st.session_state["sim_method"] == "OpenAI Embeddings" and not api_key:
        st.warning("Please enter and verify your OpenAI API key above to use OpenAI Embeddings.")
        st.stop()
    suggestions = suggest_mapping(headers, CANONICAL_SCHEMA, promoted_map=promoted)

    # Reset user_mapping if sim_method changed or mapping is None
    if (
        "user_mapping" not in st.session_state
        or st.session_state.user_mapping is None
        or "last_sim_method" not in st.session_state
        or st.session_state.last_sim_method != st.session_state["sim_method"]
    ):
        st.session_state.user_mapping = {s[0]: s[1] for s in suggestions}
        st.session_state.last_sim_method = st.session_state["sim_method"]

    st.write("Suggested mappings (you can override):")
    mapping_cols = {}
    cols = st.columns((2, 3, 1, 2))
    cols[0].markdown("**Source Column**")
    cols[1].markdown("**Suggested Canonical Field**")
    cols[2].markdown("**Confidence**")

    if "user_mapping" not in st.session_state or st.session_state.user_mapping is None:
        st.session_state.user_mapping = {s[0]: s[1] for s in suggestions}

    for src, suggested, conf, how in suggestions:
        c0, c1, gap, c2 = st.columns((2, 3, 1, 2))
        c0.write(f"**{src}**")
        options = ["(ignore)"] + CANONICAL_SCHEMA
        sel = c1.selectbox(f"map_{src}", options, index=(options.index(st.session_state.user_mapping.get(src)) if st.session_state.user_mapping.get(src) in options else 0), key=f"map_{src}_key")
        # c2.write(f"{conf:.0f}% {'(promoted)' if how=='promoted' else ''}")
        progress = int(conf)
        c2.progress(progress / 100)
        c2.caption(f"{conf:.0f}% {'(promoted)' if how=='promoted' else ''}")
        if sel == "(ignore)":
            st.session_state.user_mapping[src] = None
        else:
            st.session_state.user_mapping[src] = sel

    targets = [v for v in st.session_state.user_mapping.values() if v]
    dup_targets = {t for t in targets if targets.count(t) > 1}
    if dup_targets:
        st.warning(f"Multiple source columns mapped to same canonical target: {', '.join(dup_targets)}")
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