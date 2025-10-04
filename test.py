import os, time, traceback
import streamlit as st
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv
import torch, torch.nn as nn
import hashlib, unicodedata, math


# =========================
# ----------- UI ----------
# =========================
st.title("🧠 Conversational Neo4j Assistant")

colA,colB,colC = st.columns([1,1,2])
with colA:
    if st.button("🔌 Test Neo4j"):
        try:
            start=time.time(); rows=run_tx("RETURN 1 as ok") or []
            ok=rows[0].get('ok') if rows else "N/A"
            st.success(f"Neo4j OK ({ok}) — {time.time()-start:.3f}s")
        except Exception as e:
            st.error(f"Neo4j lỗi: {e}")
with colB:
    st.write(f"DEBUG: {'✅' if DEBUG else '❌'}")
    st.write(f"Render history: {'✅' if RENDER_HISTORY else '❌'}")
with colC:
    with st.expander("🔎 Env / Driver"):
        st.write(f"- NEO4J_URI: {'✅' if NEO4J_URI else '❌'}")
        st.write(f"- NEO4J_USER: {'✅' if NEO4J_USER else '❌'}")
        st.write(f"- NEO4J_PASSWORD: {'✅' if NEO4J_PASSWORD else '❌'}")
        st.write(f"- MOE_GATE_PATH: {'✅' if os.path.exists(GATE_PATH) else '❌'}")

st.markdown("---")

user_input=st.text_input("Nhập câu hỏi", placeholder="VD: Triệu chứng của viêm âm đạo?")

if user_input:
    try:
        ans, cy, rows, probs_dict, dur, expert, ctx_flag, anchor = query_graph(user_input)
        st.subheader("🧠 Trả lời")
        st.write(ans)
        st.caption(f"⏱ {dur:.3f}s")
        st.subheader("📊 MoE Probabilities")
        st.json(probs_dict)
        st.subheader("🔮 Expert chọn")
        st.write(expert)
        st.subheader("⚑ Context flag")
        st.write(ctx_flag)
        st.subheader("🎯 Anchor detection")
        st.json(anchor)
        if DEBUG and cy:
            st.subheader("🔧 Cypher đã chạy")
            st.code(cy, language="cypher")
            st.json(rows)
    except Exception as e:
        st.error(f"❌ Failed: {e}")
        if DEBUG: st.code(traceback.format_exc(), language="text")