import streamlit as st
import requests
import time

API_URL = "https://aimrs-research-backend.hf.space"

st.set_page_config(page_title="AI Research Assistant", layout="wide")

# -----CSS ---------
st.markdown("""
<style>
.main-title { text-align:center; font-size:40px; font-weight:700; margin-bottom:5px; }
.sub-title { text-align:center; color:#bbbbbb; margin-bottom:30px; }

.preview-wrapper { display:flex; justify-content:center; margin-top:20px; }
.preview-box {
    width:75%;
    height:550px;
    overflow-y:auto;
    padding:20px;
    border-radius:14px;
    border:2px solid #4CAF50;
    background-color:#0e1117;
    font-size:15px;
    line-height:1.7;
    color:white;
}

.download-card {
    width:70%;
    margin:auto;
    padding:25px;
    border-radius:14px;
    background-color:#161b22;
    border:1px solid #30363d;
    margin-top:30px;
}

.history-card {
    padding:12px;
    border-radius:10px;
    border:1px solid #30363d;
    margin-bottom:8px;
    background-color:#0e1117;
}

.stepper {
    display:flex;
    justify-content:center;
    align-items:center;
    margin-top:20px;
    margin-bottom:20px;
}

.step {
    display:flex;
    align-items:center;
    font-weight:600;
}
.circle {
    width:34px;
    height:34px;
    border-radius:50%;
    background:#444;
    color:white;
    display:flex;
    justify-content:center;
    align-items:center;
    margin-right:8px;
}
.active { background:#4CAF50 !important; }
.line {
    width:80px;
    height:4px;
    background:#444;
    margin:0 10px;
}
.line.active { background:#4CAF50; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>AI Research Paper Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Multi-Agent IEEE Research Paper Generator</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SESSION STATE ----------------
if "current_markdown" not in st.session_state:
    st.session_state.current_markdown = None
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "history" not in st.session_state:
    st.session_state.history = []
if "progress_step" not in st.session_state:
    st.session_state.progress_step = 0

# ---------------- LOAD HISTORY FROM BACKEND ----------------
if st.button("🔄 Load Report History"):
    try:
        res = requests.get(f"{API_URL}/history", timeout=30)
        st.session_state.history = res.json().get("history", [])
        st.success("History loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load history: {str(e)}")

# ---------------- INPUT ----------------
topic = st.text_input(" Enter Research Topic", placeholder="topic...")
run_btn = st.button("🚀 Run Research")

# ---------------- STEPPER PROGRESS TRACKER ----------------
def render_stepper(step):
    steps = ["Retrieval", "Summary", "Evaluation", "Design", "Report"]
    html = "<div class='stepper'>"

    for i, s in enumerate(steps, start=1):
        active = "active" if step >= i else ""
        symbol = "✔" if step > i else i

        html += f"""
        <div class='step'>
            <div class='circle {active}'>{symbol}</div>{s}
        </div>
        """

        if i < len(steps):
            html += f"<div class='line {active}'></div>"

    html += "</div>"

    stepper_placeholder.markdown(html, unsafe_allow_html=True)

stepper_placeholder = st.empty()
# ---------------- RUN PIPELINE WITH LIVE TRACKING ----------------
if run_btn:
    st.session_state.progress_step = 1
    render_stepper(1)

    if not topic.strip():
        st.warning("⚠️ Please enter a research topic.")
    else:
        try:
            with st.spinner("Running full pipeline..."):
                res = requests.post(
                    f"{API_URL}/run_pipeline",
                    json={"topic": topic},
                    timeout=300
                )
                data = res.json()

            if data.get("status") != "ok":
                st.error("Pipeline failed")
                st.json(data)
                st.stop()

            session_id = data["session_id"]

            while st.session_state.progress_step < 6:
                time.sleep(1)

                p = requests.get(
                    f"{API_URL}/progress/{session_id}",
                    timeout=15
                ).json()

                step = p.get("step", 1)
                st.session_state.progress_step = step
                render_stepper(step)

                if step >= 6:
                    break

            st.success("Research Completed Successfully!")

            md_content = data["report_markdown"]["content"]

            record = {
                "topic": topic,
                "markdown": md_content
            }
            st.session_state.history.insert(0, record)

            st.session_state.current_markdown = md_content
            st.session_state.current_topic = topic

        except Exception as e:
            st.error(str(e))
            
# ---------------- COLLAPSIBLE HISTORY PANEL ----------------
st.markdown("---")

with st.expander("Previous Reports", expanded=False):

    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            with st.container():
                col1, col2 = st.columns([5, 2])

                with col1:
                    st.markdown(
                        f"<div class='history-card'>📄 <b>{item['topic']}</b></div>",
                        unsafe_allow_html=True
                    )

                with col2:
                    if st.button("👁 Preview", key=f"preview_{idx}"):
                        st.session_state.current_markdown = item["markdown"]
                        st.session_state.current_topic = item["topic"]
    else:
        st.info("No previous reports yet.")
        
# ---------------- PREVIEW ----------------
if st.session_state.current_markdown:
    st.markdown("---")
    st.markdown("<h2 style='text-align:center;'>📄 Research Paper Preview</h2>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="preview-wrapper">
            <div class="preview-box">
                {st.session_state.current_markdown.replace("\n", "<br>")}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- DOWNLOAD PANEL ----------------
if st.session_state.current_markdown:
    st.subheader("⬇️ Download Report")

    col1, col2 = st.columns(2)

    with col1:
        md_res = requests.post(
            f"{API_URL}/download",
            json={"topic": st.session_state.current_topic},
            timeout=120
        )

        st.download_button(
            label="📄 Download Markdown",
            data=md_res.content,
            file_name=f"{st.session_state.current_topic.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        zip_res = requests.post(
            f"{API_URL}/download-zip",
            json={"topic": st.session_state.current_topic},
            timeout=120
        )

        st.download_button(
            label="📦 Download Overleaf ZIP",
            data=zip_res.content,
            file_name=f"{st.session_state.current_topic.replace(' ', '_')}_report.zip",
            mime="application/zip",
            use_container_width=True
        )
