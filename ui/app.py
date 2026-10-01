"""Local Streamlit demo for the HRE-TRANSLATE FastAPI service."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("HRE_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api_call(method: str, path: str, payload: dict | None = None) -> object:
    with httpx.Client(timeout=300.0) as client:
        response = client.request(method, f"{API_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def main() -> None:
    st.set_page_config(page_title="H’rê → Tiếng Việt", page_icon="🌿", layout="wide")
    st.title("H’rê → Tiếng Việt")
    st.caption("Demo nghiên cứu · từ điển, Translation Memory và NLLB + LoRA")

    try:
        health = api_call("GET", "/health")
        models = api_call("GET", "/models")
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"Không kết nối được API tại {API_URL}: {exc}")
        st.info("Hãy chạy FastAPI trước, sau đó tải lại trang này.")
        return

    if not health["neural_available"]:
        st.warning("NLLB chưa sẵn sàng trên máy này. Từ điển và Translation Memory vẫn dùng được.")
    choices = {item["id"]: item["label"] for item in models}
    with st.form("translate_form"):
        source = st.text_area("Nhập câu H’rê", height=140, max_chars=2000)
        model = st.selectbox("Mô hình", options=list(choices), format_func=choices.get)
        submitted = st.form_submit_button("Dịch", type="primary")
    if submitted:
        st.session_state.pop("translation_result", None)
        try:
            result = api_call(
                "POST",
                "/translate",
                {"text": source, "source": "hre", "target": "vi", "model": model},
            )
            st.session_state["translation_result"] = result
            st.session_state["source_text"] = source
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("detail", str(exc))
            st.error(f"Không thể dịch: {detail}")
        except (httpx.HTTPError, ValueError) as exc:
            st.error(f"Không thể dịch: {exc}")

    result = st.session_state.get("translation_result")
    if result:
        st.subheader("Bản dịch")
        st.write(result["translation"])
        st.caption(f"Mô hình: {result['model']} · Độ trễ: {result['latency_ms']:.2f} ms")
        st.subheader("Từ điển khớp")
        terms = result.get("retrieved_terms", [])
        st.dataframe(terms, hide_index=True) if terms else st.write("Không có mục từ khớp.")
        st.subheader("Ví dụ truy hồi")
        examples = result.get("retrieved_examples", [])
        st.dataframe(examples, hide_index=True) if examples else st.write("Không có ví dụ.")

        with st.form("feedback_form"):
            correction = st.text_area("Bạn muốn sửa bản dịch như thế nào?", height=100)
            rating = st.selectbox(
                "Đánh giá (tùy chọn)",
                options=[None, 1, 2, 3, 4, 5],
                format_func=lambda value: "Không đánh giá" if value is None else f"{value}/5",
            )
            feedback_submitted = st.form_submit_button("Gửi góp ý")
        if feedback_submitted:
            try:
                saved = api_call(
                    "POST",
                    "/feedback",
                    {
                        "source": st.session_state["source_text"],
                        "prediction": result["translation"],
                        "correction": correction,
                        "model": result["model"],
                        "rating": rating,
                    },
                )
                st.success(f"Đã lưu góp ý #{saved['id']}. Cảm ơn bạn!")
            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get("detail", str(exc))
                st.error(f"Không lưu được góp ý: {detail}")
            except (httpx.HTTPError, ValueError) as exc:
                st.error(f"Không lưu được góp ý: {exc}")


if __name__ == "__main__":
    main()
