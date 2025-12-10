import streamlit as st
import pandas as pd
import base64

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Từ điển H're - Việt",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. KHAI BÁO BIẾN ---
DATA_FILE = 'data.csv'
IMAGE_FILE = 'anhbia.jpg'

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---
@st.cache_data
def load_data():
    data = []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.lower().startswith('hre,vietnamese'): continue
            parts = line.split(',', 1) 
            if len(parts) >= 2:
                hre_word = parts[0].strip()
                viet_word = parts[1].strip().replace('"', '')
                data.append([hre_word, viet_word])
        return pd.DataFrame(data, columns=['hre', 'vietnamese'])
    except Exception:
        return pd.DataFrame(columns=['hre', 'vietnamese'])

def save_data(new_hre, new_viet):
    try:
        if "," in new_viet: new_viet = f'"{new_viet}"'
        line = f"\n{new_hre},{new_viet}"
        with open(DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(line)
        st.cache_data.clear()
        return True
    except Exception:
        return False

# --- 4. XỬ LÝ ẢNH NỀN ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return ""

img_base64 = get_base64_image(IMAGE_FILE)

# --- 5. CSS (GIAO DIỆN SINH ĐỘNG) ---
st.markdown(f"""
    <style>
    /* Import Font đẹp từ Google */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=Open+Sans:wght@400;600&display=swap');

    /* --- TỔNG THỂ --- */
    html, body, [class*="css"] {{
        font-family: 'Open Sans', sans-serif;
    }}
    
    /* Nền trang Gradient Lạnh nhưng Tươi (Xanh ngọc -> Tím hồng nhạt) */
    .stApp {{
        background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%);
        background-attachment: fixed;
    }}

    /* --- CONTAINER CHÍNH --- */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.85); /* Hiệu ứng kính mờ */
        backdrop-filter: blur(10px); /* Làm mờ nền phía sau */
        border-radius: 20px;
        max-width: 1000px;
        padding-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        /* Padding bottom lớn để chứa footer không bị che */
        padding-bottom: 60px !important; 
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
    }}

    /* --- HEADER ẤN TƯỢNG --- */
    .header-box {{
        position: relative;
        height: 250px;
        background-image: url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        border-radius: 20px 20px 0 0; /* Bo góc trên */
        overflow: hidden;
    }}
    
    .header-gradient {{
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        /* Lớp phủ màu tối dần xuống dưới để làm nổi chữ */
        background: linear-gradient(to bottom, rgba(63, 43, 150, 0.2), rgba(255, 255, 255, 1));
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding-top: 40px;
    }}

    /* Tiêu đề chữ Gradient (Màu chuyển sắc) */
    .title-text {{
        font-family: 'Montserrat', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-transform: uppercase;
        
        /* Hiệu ứng chữ chuyển màu */
        background: -webkit-linear-gradient(45deg, #3f2b96, #2F80ED, #00C9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        
        /* Bóng nhẹ để nổi bật trên nền */
        filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.2));
    }}
    
    .subtitle-text {{
        font-family: 'Montserrat', sans-serif;
        color: #555;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 5px;
        background: rgba(255,255,255,0.6);
        padding: 5px 20px;
        border-radius: 30px;
    }}

    /* --- TABS --- */
    .stTabs {{
        margin-top: 20px;
        padding: 0 30px;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        justify-content: center;
        gap: 15px;
        border: none;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        border-radius: 30px;
        padding: 10px 25px;
        color: #666;
        font-weight: 700;
        border: 1px solid #eee;
        transition: all 0.3s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        transform: translateY(-2px);
        color: #2F80ED;
        border-color: #2F80ED;
    }}

    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(90deg, #2F80ED 0%, #56CCF2 100%);
        color: white;
        border: none;
        box-shadow: 0 5px 15px rgba(47, 128, 237, 0.3);
    }}

    /* --- INPUT --- */
    .stTextInput input {{
        border-radius: 12px;
        padding: 12px 15px;
        border: 2px solid #e0e0e0;
        font-size: 1.1rem;
        transition: border 0.3s;
    }}
    .stTextInput input:focus {{
        border-color: #2F80ED;
        box-shadow: 0 0 0 4px rgba(47, 128, 237, 0.1);
    }}

    /* --- KẾT QUẢ (Card) --- */
    .result-card {{
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 6px solid #2F80ED; /* Viền trái màu xanh */
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        transition: transform 0.2s, box-shadow 0.2s;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    
    .result-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(47, 128, 237, 0.15); /* Bóng xanh khi hover */
    }}

    .hre-text {{
        font-size: 1.4rem;
        font-weight: 800;
        color: #3f2b96; /* Tím đậm */
    }}
    
    .viet-text {{
        font-size: 1.2rem;
        color: #333;
        font-weight: 500;
    }}

    /* --- FOOTER (Quan trọng: Padding và Margin để không bị che) --- */
    .footer-wrapper {{
        margin-top: 50px;
        padding: 30px 20px;
        text-align: center;
        background-color: #f9fbfc;
        border-top: 1px solid #eee;
        border-radius: 0 0 20px 20px;
    }}
    
    .footer-title {{
        font-weight: 800;
        color: #3f2b96;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .footer-info {{
        color: #666;
        margin-top: 5px;
        font-size: 0.9rem;
        line-height: 1.6;
    }}

    /* Ẩn bớt UI Streamlit */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    
    </style>
""", unsafe_allow_html=True)

# --- 6. GIAO DIỆN CHÍNH ---

def main():
    # HEADER với Background Image & Text Gradient
    st.markdown(f"""
        <div class="header-box">
            <div class="header-gradient">
                <h1 class="title-text">TỪ ĐIỂN H'RE</h1>
                <div class="subtitle-text">Tra cứu nhanh chóng - Đóng góp dễ dàng</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    df = load_data()

    # NỘI DUNG CHÍNH (Thêm padding để nội dung không dính sát lề)
    st.markdown('<div style="padding: 0 40px;">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔍 TRA CỨU", "ĐÓNG GÓP"])

    # --- TAB TRA CỨU ---
    with tab1:
        st.write("")
        col_space1, col_center, col_space2 = st.columns([1, 4, 1])
        with col_center:
            search = st.text_input("Tra từ:", placeholder="Nhập từ H're hoặc tiếng Việt...", label_visibility="collapsed")
            
            if search:
                s_lower = search.lower()
                results = df[
                    df['hre'].str.lower().str.contains(s_lower, na=False) | 
                    df['vietnamese'].str.lower().str.contains(s_lower, na=False)
                ]
                
                st.markdown(f"<p style='text-align:center; color:#2F80ED; margin: 15px 0; font-weight:600;'>🎉 Tìm thấy {len(results)} kết quả</p>", unsafe_allow_html=True)

                if not results.empty:
                    for _, row in results.iterrows():
                        st.markdown(f"""
                            <div class="result-card">
                                <div>
                                    <div style="font-size:0.8rem; color:#888;">TIẾNG H'RE</div>
                                    <div class="hre-text">{row['hre']}</div>
                                </div>
                                <div style="font-size:1.5rem; color:#ddd;">➝</div>
                                <div style="text-align:right;">
                                    <div style="font-size:0.8rem; color:#888;">TIẾNG VIỆT</div>
                                    <div class="viet-text">{row['vietnamese']}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Chưa tìm thấy từ này. Hãy thử từ khác xem sao!")
            else:
                 st.markdown("""
                    <div style="text-align:center; padding: 40px; opacity: 0.6;">
                        <span style="font-size: 3rem;"></span>
                        <p>Nhập từ khoá để bắt đầu hành trình khám phá ngôn ngữ.</p>
                    </div>
                """, unsafe_allow_html=True)

    # --- TAB ĐÓNG GÓP ---
    with tab2:
        st.write("")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("""
                <div style="text-align:center; margin-bottom:20px;">
                    <h3 style="color:#3f2b96;">Thêm từ mới</h3>
                    <p style="color:#666;">Cảm ơn bạn đã chung tay bảo tồn ngôn ngữ H're</p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.form("contribute"):
                new_hre = st.text_input("Từ H're:")
                new_viet = st.text_input("Nghĩa Tiếng Việt:")
                
                # Nút bấm style Gradient
                btn = st.form_submit_button("Lưu Đóng Góp", use_container_width=True)
                
                if btn:
                    if new_hre and new_viet:
                        save_data(new_hre, new_viet)
                        st.success("Tuyệt vời! Dữ liệu đã được lưu.")
                        df = load_data()
                    else:
                        st.error("Đừng để trống ô nào nhé!")

    st.markdown('</div>', unsafe_allow_html=True) # Đóng div padding nội dung

    # --- FOOTER (Được đặt trong div riêng biệt, padding an toàn) ---
    st.markdown("""
        <div class="footer-wrapper">
            <div class="footer-title">Dự án Từ điển H're - Việt</div>
            <div class="footer-info">
                Phát triển bởi <b>Huỳnh Thanh Khải</b><br>
                Email: huynhthanhkhaibato2011@gmail.com 
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()

