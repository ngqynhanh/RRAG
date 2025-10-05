# synth_qas_pipeline_vi.py
"""
Pipeline tạo dataset Q&A (Tiếng Việt)
- Input:
  * datasets/graph_info.txt  (mỗi dòng 1 relation chứa {name: ..., layer: ...})
  * datasets/passages.jsonl  (mỗi dòng JSON: {"id":..., "text":..., "meta": {...}})
- Output:
  * datasets/qas_synthetic_vi.jsonl (mỗi dòng JSON: {"question":..., "answers":[...], "ctxs":[...]} )
- Tùy chọn LLM: OpenRouter (OPENROUTER_API_KEY) hoặc local HF (USE_LOCAL_MODEL, LOCAL_MODEL_ID)
"""
import os, re, json, random, textwrap, logging
from pathlib import Path
from typing import List, Optional, Dict, Set

# load .env nếu có
from dotenv import load_dotenv
load_dotenv()

# ---------- CẤU HÌNH ----------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_ID = os.getenv("MODEL_ID", "qwen/qwen-2.5-72b-instruct")   # dùng với OpenRouter
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "0") in ("1", "true", "True")
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "LLM360/K2-Think")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", "D:/huggingface_cache")

PASSAGES_PATH = Path(os.getenv("PASSAGES_PATH", "datasets/passages.jsonl"))
GRAPH_INFO_PATH = Path(os.getenv("GRAPH_INFO_PATH", "datasets/graph_info.txt"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "datasets/qas_synthetic.jsonl"))

TOP_K_PASSAGES = int(os.getenv("TOP_K_PASSAGES", "5"))
NEGATIVE_CTXS = int(os.getenv("NEGATIVE_CTXS", "3"))
MAX_SENTENCES_PER_ANSWER = int(os.getenv("MAX_SENTENCES_PER_ANSWER", "6"))
REFINE_ANSWER_WITH_LLM = os.getenv("REFINE_ANSWER_WITH_LLM", "1") in ("1", "true", "True")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_Q_PER_ENTITY = int(os.getenv("MAX_Q_PER_ENTITY", "4"))

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- LLM / local / sbert clients ----------
client = None
local_generator = None
sbert_model = None
_USE_SBERT = False

# Try OpenRouter/OpenAI client if key present
if OPENROUTER_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        logging.info("OpenRouter client ready.")
    except Exception as e:
        logging.warning("Không thể khởi tạo OpenRouter client: %s", e)
        client = None

# # Local HF generator (optional)
# if USE_LOCAL_MODEL:
#     try:
#         os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
#         from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
#         import torch
#         tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID, cache_dir=TRANSFORMERS_CACHE, use_fast=True)
#         model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID, cache_dir=TRANSFORMERS_CACHE, torch_dtype="auto")
#         device = 0 if torch.cuda.is_available() else -1
#         local_generator = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
#         logging.info("Loaded local HF model %s (device=%s).", LOCAL_MODEL_ID, device)
#     except Exception as e:
#         logging.warning("Không thể load local model: %s", e)
#         local_generator = None

# # SBERT fallback (optional)
# try:
#     from sentence_transformers import SentenceTransformer, util as sbert_util
#     sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     _USE_SBERT = True
#     logging.info("SBERT ready.")
# except Exception:
#     _USE_SBERT = False

# ---------- Templates / vocab (từ bạn) ----------
ALL_FIELDS=[
    "Disease","SubDisease","Topic","SubTopic",
    "Complication","Treatment","Application","Cause",
    "Definition","Detail","Population","Advice",
    "RiskFactor","Prevention","Symptom"
]
IDX={k:i for i,k in enumerate(ALL_FIELDS)}
NUM_CLASSES=len(ALL_FIELDS)

DISEASES=["Đau bụng kinh","Vô kinh","Hội chứng buồng trứng đa nang","Hội chứng buồng trứng đa nang ","U xơ tử cung","Lạc nội mạc tử cung","Bệnh viêm vùng chậu","Viêm âm đạo","Bệnh khí hư","Đau khi quan hệ tình dục","Loạn khuẩn âm đạo","Bệnh vú lành tính","Tuyến vú lành tính","Xơ nang tuyến vú","Hemophilia","Bệnh Huntington","Monosomy","Khiếm khuyết ống thần kinh","Trisomy 13","Trisomy 18","Trisomy 21","Hội chứng Turner","Polyp cổ tử cung","Viêm gan B","Viêm gan C","Viêm gan siêu vi","Viêm màng ối","Rubella","Giang mai","Chlamydia","Lậu","Lao","Dị tật bẩm sinh","Khuyết tật ống thần kinh","Khuyết tật thành bụng","Tim bẩm sinh","Hội chứng Down","Hội chứng tam nhiễm sắc thể 18 (hội chứng Edward)","Huyết áp cao","Cao huyết áp mãn tính","Tiền sản giật","Tiền sản giật chồng cao huyết áp mãn tính","Thuỷ đậu","Bệnh Herpes sinh dục","Bệnh lây truyền qua đường tình dục (STDs)","Bệnh lây truyền qua đường tình dục","HIV","Liên cầu khuẩn nhóm B","Tiểu đường","Tiền tiểu đường","Sinh non","Sinh cực non","Động kinh","Xơ gan","Hội chứng truyền máu song thai (TMST)","Bệnh xơ nang","Bất đồng nhóm máu Rh","Lộ tuyến cổ tử cung","Chảy máu tử cung bất thường","Tiểu không kiểm soát khi gắng sức","Tiểu không tự chủ","Huyết khối tĩnh mạch sâu","Tăng sinh nội mạc tử cung","Nhiễm trùng đường niệu","Viêm nang lông","Viêm da tiếp xúc","Nang tuyến Bartholin","Địa y simplex chronicus","Địa y sclerosus","Địa y planus","Đau âm hộ mãn tính","Teo âm hộ","Tân sinh trong biểu mô âm hộ (VIN)","Ung thư âm hộ","Đau bụng dưới kinh niên","Hội chứng tiền kinh nguyệt","Xuất huyết tử cung bất thường","Ung thư buồng trứng","Ung thư tử cung","U nang buồng trứng","Viêm vùng chậu","Nhiễm trùng virus Papilloma ở người (HPV)","SIDS","Hội chứng FAS","Béo phì","Dị tật ống thần kinh","Trứng trống (thai không phôi)","Thai lưu","Hở eo cổ tử cung","Thai trứng","Suy cổ tử cung","Hội chứng kháng phospholipid (APS - hội chứng Hughes)","Hội chứng kháng phospholipid","Ối vỡ non trên thai non tháng (PPROM)","Vàng da sơ sinh","Vàng da do sữa mẹ","Vàng da do không bú mẹ","Serratia marcescens","Cổ tử cung ngắn","Ngưng thở ở trẻ sinh non","Hội chứng suy hô hấp (RDS) ở trẻ sinh non","Xuất huyết não thất (IVH) ở trẻ sinh non","Còn ống động mạch (PDA) ở trẻ sinh non","Viêm ruột hoại tử (NEC) ở trẻ sinh non","Bệnh võng mạc ở trẻ sinh non (ROP)","Vàng da ở trẻ sinh non","Thiếu máu ở trẻ sinh non","Loạn sản phế quản – phổi (BPD) ở trẻ sinh non","Nhiễm trùng ở trẻ sinh non","Virus Zika","Teo âm đạo","Khô âm đạo","Hội chứng tiết niệu sinh dục thời kỳ mãn kinh (GSM)","Bệnh động mạch vành","Hội chứng Fitz-Hugh-Curtis","Hội chứng Fitz-Hugh-Curtis","Viêm túi mật","Hoại tử mỡ tuyến vú","Ung thư dạ dày di truyền thể khuếch tán","Bệnh chàm","Viêm tiểu phế quản ở trẻ"]

SUBDISEASES=["Đau bụng kinh nguyên phát","Đau bụng kinh thứ phát","Chứng kinh nguyệt ẩn","Vô kinh nguyên phát","Vô kinh thứ phát","Viêm âm đạo do trùng roi","Viêm âm đạo không nhiễm trùng","Viêm âm đạo do Trichomonas","Viêm âm đạo do nấm Candida","Viêm teo âm đạo","Nhiễm nấm âm đạo","Nhiễm khuẩn âm đạo","Đau vú theo chu kỳ","Đau vú không theo chu kỳ","Viêm vú","Tiết dịch núm vú","U nang tuyến vú","Bướu sợi tuyến","Viêm tuyến vú","Viêm gan B cấp tính","Viêm gan B mãn tính","Chlamydia","Bệnh lậu","Bệnh giang mai","Tăng huyết áp mãn tính","Hội chứng HELLP","AIDS","HIV","Tiểu đường loại 1","Tiểu đường loại 2","Bệnh tán huyết","Thuyên tắc phổi","Tăng sinh nội mạc tử cung đơn thuần","Tăng sinh nội mạc tử cung phức tạp","Tăng sinh không điển hình đơn thuần","Tăng sinh không điển hình phức tạp","Rối loạn tâm thần tiền kinh nguyệt","Sarcoma tử cung","Ung thư nội mạc tử cung","Đau âm hộ mãn tính toàn bộ","Đau âm hộ mãn tính khu trú","Viêm túi mật cấp tính","Viêm túi mật mãn tính"]

SYMPTOMS=[
  "mệt mỏi","đau khi quan hệ tình dục","ngứa vùng kín","khí hư có mùi hôi",
  "ra máu sau quan hệ","đau vùng chậu","tiểu buốt","tiểu rát","bốc hỏa",
  "mất ngủ","rối loạn kinh nguyệt","trễ kinh nhiều ngày","đau bụng kinh dữ dội",
  "khó chịu khi tiểu","chảy máu âm đạo bất thường","đau lưng dưới",
  "đau bụng dưới","khô âm đạo","đi tiểu nhiều lần","đau rát âm hộ"
]

TOPICS=["Rối loạn phụ khoa","Mãn kinh","Tầm quan trọng của việc khám phụ khoa","Lời khuyên sức khỏe","Gen","Nhiễm sắc thể","Yếu tố quyết định giới tính bé","Rối loạn di truyền","Người lành mang gen bệnh","Nguy cơ cao sinh con dị tật bẩm sinh","Tư vấn di truyền","Xét nghiệm tiền sản","Sàng lọc trước sinh","Xét nghiệm phát hiện người mang gen bệnh","Đối tượng xét nghiệm mang gen bệnh","Thời điểm xét nghiệm mang gen bệnh","Xét nghiệm chẩn đoán trẻ dị tật bẩm sinh","Rủi ro xét nghiệm chẩn đoán","Bao cao su","Thuốc tránh thai","Que cấy tránh thai","Viên uống tránh thai kết hợp","Phương pháp tránh thai có rào cản","Thuốc diệt tinh trùng","Miếng xốp tránh thai","Màng ngăn âm đạo","Mũ chụp cổ tử cung","Thuốc tiêm tránh thai","Phương pháp ngừa thai nội tiết phối hợp","Viên uống nội tiết phối hợp","Vòng âm đạo","Miếng dán tránh thai","Ngừa thai khẩn cấp","Ngừa thai khẩn cấp IUD","Biến chứng thai kỳ","Rau tiền đạo","Rau cài răng lược","Mạch máu tiền đạo","Máu báo thai","Nhau bong non","Nhau tiền đạo","Nhau cài răng lược","Sinh non","Cử động thai","Xét nghiệm chẩn đoán có thai","Siêu âm ngã âm đạo","Siêu âm","Soi cổ tử cung","Công thức máu","Nhóm máu","Xét nghiệm nước tiểu","Cấy nước tiểu","Kháng thể Rh","Xét nghiệm dung nạp đường","GBS","Sàng lọc dị tật thai","Chẩn đoán dị tật thai","Người mang gen","Dị tật bẩm sinh","Xét nghiệm theo dõi sức khỏe thai nhi","Đếm cử động thai","Test không đả kích","Trắc đồ sinh vật lý","Trắc đồ sinh vật lý cải tiến","Test co thắt đả kích","Xét nghiệm sàng lọc","Khuyết tật ống thần kinh","Khuyết tật thành bụng","Tim bẩm sinh","Hội chứng Down","Hội chứng Edwards","Đối tượng sàng lọc","Xét nghiệm sàng lọc quý I","PAPP-A","hCG","Độ mờ da gáy","Xét nghiệm sàng lọc quý II","AFP","Estriol","Inhibin A","Sàng lọc kết hợp","Xét nghiệm chẩn đoán","Chẩn đoán tiền sản","X quang","Thai ngoài tử cung","Bệnh lây truyền qua đường tình dục","Ngôi mông","Ngoại xoay thai (ECV)","Ngày sinh dự kiến","Thai quá ngày","Theo dõi tim thai bằng thiết bị điện tử","Thử nghiệm không áp lực","Đo chỉ số sinh lý học","Thử nghiệm có cơn co tử cung","Làm mềm cổ tử cung","Bóc tách màng ối","Bấm ối","Sử dụng oxytocin","Nguy cơ khi sinh con muộn","Khả năng sinh sản","Nguy cơ bệnh lý khi sinh con muộn","Huyết áp cao","Xét nghiệm sàng lọc và chẩn đoán","Mang đa thai","Nguy cơ của đa thai","Mổ lấy thai","Chuẩn bị trước khi mang thai","Chăm sóc khi mang thai","Máu dây rốn","Tế bào gốc","Tế bào gốc máu dây rốn","Ngân hàng máu dây rốn","Ngân hàng máu dây rốn cộng đồng","Ngân hàng máu dây rốn cá nhân","Lấy máu dây rốn","Song thai","Sẩy thai","Thai lạc chỗ","Rau bong non","Xét nghiệm tầm soát người mang gen","Viêm vú sau sinh","Estrogen","Liệu pháp hormone","Progestin","Mô tuyến vú","Phát hiện khối u ở vú","Vô sinh","Khám vô sinh","Vaccine chống virus papilloma ở người (HPV)","Các biện pháp tránh thai nội tiết","Cấy que tránh thai","Tiêm tránh thai","Triệt sản","Thuốc tránh thai kết hợp","Thuốc tránh thai chỉ chứa progestin","Các dụng cụ rào cản tránh thai","Chất diệt tinh trùng","Xốp đệm tránh thai","Màng chắn tránh thai","Dụng cụ tử cung","Phương pháp tránh thai tự nhiên","Sự rụng trứng","Phương pháp theo dõi nhiệt độ cơ thể","Phương pháp theo dõi dịch nhầy cổ tử cung","Phương pháp nhiệt chứng","Phương pháp theo dõi lịch","Phương pháp mất kinh do tạo sữa","Thắt ống dẫn trứng","Tránh thai bằng màng ngăn âm đạo","Tránh thai khẩn cấp","Lựa chọn biện pháp tránh thai","Kế hoạch hoá gia đình bằng phương pháp tự nhiên","Bài tập cần tránh khi mang thai","Ốm nghén","Sức khỏe răng miệng và sức khỏe toàn thân","Thử thai","Luật chống phân biệt thông tin di truyền","Mang thai","Mức tăng cân khuyến cáo","Tỷ lệ tăng cân phù hợp","Nguy cơ liên quan đến cân nặng","Phương pháp giữ cân nặng phù hợp","Nghiện ma túy","Chuyển dạ","Chuyển dạ giả","Giảm đau vùng và gây tê vùng","Gây tê tủy sống – ngoài màng cứng kết hợp","Chọc ối nhân tạo (ARM)","Truyền oxytocin","Rặn đẻ","Vàng da","Thính chẩn","Nhịp tim thai bất thường","Chỉ số Bishop","Làm chín muồi cổ tử cung","Prostaglandin","Lóc ối (làm tách màng ối)","Vỡ màng ối","Tia ối","Oxytocin","Liệu pháp tâm lý","Sinh theo chỉ định y khoa","Sinh theo ý muốn","Thời gian thai kì","Sự phát triển của thai nhi trong những tuần cuối","Nguy cơ trẻ sinh trước 39 tuần","Kỹ thuật Lamaze","Phương pháp Bradley","Gây tê vùng","An thần","Sinh nở tự nhiên","Trữ máu cuống rốn","Tế bào gốc tạo máu","Thai quá ngày dự sinh","Ngừa thai","Nuôi con bằng sữa mẹ","Chăm sóc vết may tầng sinh môn","Trĩ và táo bón","Quan hệ tình dục sau sinh","Tiêu tiểu không kiểm soát","Ngoại xoay thai","Progesterone","Corticosteroids trước sinh (ACS)","Kháng sinh","Thuốc giảm co tử cung","Chuyển dạ sinh non","Khâu eo cổ tử cung","Nghỉ ngơi tại giường","Chất chủ vận thụ thể Beta - adrenergic (terbutaline)","Chẹn kênh canxi (nifedipine)","Magnesium sulfate","Thuốc kháng viêm nonsteroidal (NSAIDs) (Indomethacin)","Corticosteroids","Thuốc giảm gò tử cung","Tocolytics","Surfactant","Sự thụ thai","Nhau thai","Thai kỳ","Cảm nhận cử động thai nhi","Giới tính thai nhi","Thời kỳ phôi","Túi ối","Giai đoạn thai","Tiêm phòng cúm","Vaccine ho gà","Vaccine an toàn cho phụ nữ cho con bú","Vô sinh - hiếm muộn","Xét nghiệm vô sinh dành cho nam giới","Quá trình thụ thai","Thụ tinh trong ống nghiệm (IVF)","Khám hiếm muộn","Đếm số lần cử động thai nhi","Siêu âm Doppler động mạch rốn"]

SUBTOPICS=["Các rối loạn phụ khoa phổ biến","Triệu chứng phụ khoa thường gặp","Tiền mãn kinh","Rối loạn di truyền trội trên nhiễm sắc thể thường","Rối loạn di truyền lặn trên nhiễm sắc thể thường","Nguyên nhân gây rối loạn di truyền","Nguyên nhân gây rối loạn NST","Thể dị bội","Rối loạn di truyền liên kết nhiễm sắc thể giới tính","Rối loạn di truyền đa yếu tố","Xét nghiệm chẩn đoán trẻ dị tật bẩm sinh","Ngừa thai khẩn cấp Progestin","Ngừa thai phối hợp (Yuzpe)","Ngừa thai khẩn cấp Ulipristal","Sẩy thai sớm","Siêu âm sàng lọc","Siêu âm thường","Siêu âm nâng cao","Khởi phát chuyển dạ","Song thai khác trứng","Song thai cùng trứng","Song thai bất tương xứng","Chuyển dạ sinh non","Dụng cụ tử cung","Triệt sản ống dẫn trứng","Triệt sản qua soi tử cung","Thuốc tránh thai chỉ có Progestin","Thuốc tránh thai khẩn cấp","Dụng cụ tử cung tránh thai khẩn cấp","Dụng cụ tử cung tránh thai","Sau sẩy thai","Sẩy thai ngoài tử cung","Sẩy thai liên tiếp","Giai đoạn 3 của chuyển dạ","Nguy cơ khi khởi phát chuyển dạ","Hiệu quả khởi phát chuyển dạ","Chuyển dạ theo ý muốn","Hiến máu cuống rốn của bé","Progesterone đặt âm đạo","Progesterone tiêm","Sinh cực non","Xét nghiệm vô sinh dành cho nữ giới","Vô sinh chưa rõ nguyên nhân"]

TEMPLATES = {
  "Symptom": [
    "Triệu chứng của {main}","Dấu hiệu {main}","Biểu hiện khi mắc {main}",
    "Các dấu hiệu nhận biết {main}","Triệu chứng đặc trưng của {main}",
    "Khi nào thì biết mình đang mắc {main}","Làm sao nhận biết {main}",
    "{main} có dấu hiệu ban đầu ra sao","Có triệu chứng tiềm ẩn nào của {main} không"
  ],
  "Cause": [
    "Nguyên nhân gây {main}","Vì sao bị {main}","Điều gì gây ra {main}",
    "Lý do khiến dễ bị {main}","Nguyên nhân sâu xa dẫn tới {main}",
    "Tại sao nhiều người mắc {main}","{main} xuất hiện từ yếu tố nào",
    "Nguyên nhân phổ biến nhất của {main}","Có yếu tố môi trường nào khiến {main} không"
  ],
  "Treatment": [
    "Điều trị {main}","Cách chữa {main}","Phác đồ cho {main}",
    "Có thuốc nào dùng cho {main}","Điều trị {main} có cần phẫu thuật không",
    "Chữa {main} tại nhà có hiệu quả không","{main} điều trị bằng thuốc gì",
    "Các phương pháp y học hiện đại chữa {main}","Khi mắc {main} thì bác sĩ thường chỉ định gì"
  ],
  "Complication": [
    "Biến chứng của {main}","Hậu quả nếu bị {main}","Rủi ro lâu dài của {main}",
    "{main} nếu không chữa có nguy hiểm không","Bệnh {main} có thể dẫn đến vô sinh/ung thư không",
    "Nguy cơ về sức khỏe khi bị {main}","{main} có gây di chứng gì không",
    "Nếu {main} tiến triển nặng sẽ ra sao","{main} có làm tăng nguy cơ tử vong không"
  ],
  "Population": [
    "Ai dễ bị {main}","Đối tượng nào thường mắc {main}","{main} thường gặp ở lứa tuổi nào",
    "{main} phổ biến ở phụ nữ sau sinh không","Nhóm tuổi nào dễ mắc {main}",
    "{main} thường gặp ở nam hay nữ","Ai có khả năng mắc {main} cao nhất",
    "Đối tượng đặc biệt nào cần lưu ý về {main}","{main} có thường gặp ở trẻ em không"
  ],
  "Definition": [
    "{main} là gì","Định nghĩa {main}","Khái niệm về {main}",
    "Khái niệm cơ bản về {main}","Tóm tắt {main}","{main} có nghĩa là gì trong y học",
    "Định nghĩa ngắn gọn {main}","{main} là thuật ngữ chỉ điều gì",
    "Bạn hiểu thế nào về {main}"
  ],
  "Detail": [
    "Mô tả chi tiết về {main}","Thông tin thêm về {main}","{main}: thông tin quan trọng",
    "Những điều cần biết về {main}","Mô tả kỹ hơn {main}","Chi tiết quan trọng về {main}",
    "Cung cấp thêm thông tin về {main}","{main} bao gồm những gì",
    "Những khía cạnh cần lưu ý về {main}"
  ],
  "Subdisease": [
    "Các thể của {main}","Phân loại {main}","{main} gồm các nhóm nào",
    "Các loại {main}","Phân nhóm bệnh {main}","{main} có mấy thể chính",
    "Kể tên các dạng {main}","Bệnh {main} chia thành những thể nào",
    "Những phân nhóm đặc trưng của {main}"
  ],
  "Application": [
    "Ứng dụng của {main}","{main} được dùng để làm gì","Vai trò {main} trong y khoa",
    "{main} được dùng để làm gì trong y tế","Vai trò thực tế của {main}",
    "Ý nghĩa lâm sàng của {main}","{main} hỗ trợ bác sĩ trong việc gì",
    "Tác dụng chính của {main}","{main} thường áp dụng trong trường hợp nào"
  ],
  "Advice": [
    "Lời khuyên khi bị {main}","Nên làm gì khi có {main}","Khi {main} thì cần lưu ý gì",
    "Cách xử lý an toàn khi gặp {main}","Nếu có {main} thì cần kiêng gì",
    "Bác sĩ thường khuyên gì cho {main}","Khi {main} thì nên làm gì",
    "Khi {main} thì nên đi khám ngay không","Những điều cần tránh khi có {main}"
  ],
  "RiskFactor": [
    "Yếu tố nguy cơ của {main}","Ai có nguy cơ cao bị {main}","Tình huống nào dễ dẫn tới {main}",
    "Người nào có nguy cơ cao bị {main}","Yếu tố làm tăng khả năng mắc {main}",
    "Thói quen nào dễ gây {main}","Ai cần đi khám sớm vì nguy cơ {main}",
    "Có bệnh nền nào khiến {main} nặng hơn không","Lối sống nào làm tăng nguy cơ {main}"
  ],
  "Prevention": [
    "Cách phòng ngừa {main}","Làm sao để tránh {main}","Phòng bệnh {main} thế nào",
    "Cách tốt nhất để phòng tránh {main}","Làm thế nào để ngừa {main}",
    "Có vaccine hoặc thuốc dự phòng {main} không","Thói quen giúp hạn chế {main}",
    "Chế độ sinh hoạt nào giúp phòng tránh {main}","Thực phẩm nào giúp ngừa {main}"
  ],
  "Disease": [
    "{x} là bệnh gì","{x} là bệnh gì?","{x} thuộc bệnh nào",
    "{x} liên quan tới bệnh nào","{x} là biểu hiện của bệnh gì",
    "{x} thường gặp trong bệnh gì","{x} có thể do bệnh gì gây ra",
    "{x} là dấu hiệu của bệnh nào","Nếu bị {x} thì có thể đang mắc bệnh gì"
  ]
}

TEMPLATES_CTX={
  "Symptom":["Triệu chứng của {main} trong {ctx}","Dấu hiệu {main} khi {ctx}"],
  "Cause":["Nguyên nhân {main} trong {ctx}","Tại sao có {main} khi {ctx}"],
  "Treatment":["Điều trị {main} trong {ctx}","Xử lý {main} khi {ctx}"],
  "Complication":["Biến chứng {main} trong {ctx}","Rủi ro của {main} khi {ctx}"],
  "Population":["{main} trong {ctx} thường gặp ở ai","Ai hay mắc {main} khi {ctx}"],
  "Definition":["Định nghĩa {main} trong {ctx}","{main} là gì khi {ctx}"],
  "Detail":["Thông tin chi tiết {main} trong {ctx}","Điều cần biết về {main} khi {ctx}"],
  "Subdisease":["Các thể {main} trong {ctx}","Phân loại {main} khi {ctx}"],
  "Application":["Ứng dụng của {main} trong {ctx}","{main} được dùng thế nào khi {ctx}"],
  "Advice":["Lời khuyên khi bị {main} trong {ctx}","Nên làm gì với {main} khi {ctx}"],
  "RiskFactor":["Yếu tố nguy cơ {main} trong {ctx}","Nguy cơ mắc {main} khi {ctx}"],
  "Prevention":["Phòng ngừa {main} trong {ctx}","Làm sao tránh {main} khi {ctx}"]
}

MULTI_TEMPLATES=[
    (["Symptom","Treatment"],["Triệu chứng và cách điều trị {main}","Dấu hiệu của {main} và chữa thế nào"]),
    (["Symptom","Cause"],["Triệu chứng và nguyên nhân của {main}","Vì sao mắc {main} và có triệu chứng gì"]),
    (["Symptom","Treatment","Cause"],["Triệu chứng, nguyên nhân và cách chữa {main}","Nguyên nhân, triệu chứng và điều trị {main}"]),
    (["Definition","Population"],["{main} là gì và ai dễ mắc","Định nghĩa và đối tượng nguy cơ của {main}"]),
    (["Treatment","Complication"],["Điều trị {main} và các biến chứng có thể gặp","{main}: biến chứng và hướng điều trị"]),
    (["Cause","Population"],["Nguyên nhân và đối tượng dễ mắc {main}","{main}: vì sao bị và thường gặp ở ai"]),
    (["Subdisease","Symptom"],["Các thể {main} và triệu chứng đi kèm","Phân loại {main} cùng dấu hiệu nhận biết"]),
    (["RiskFactor","Prevention"],["Yếu tố nguy cơ và cách phòng tránh {main}","Làm sao để phòng ngừa {main} và ai dễ mắc"]),
    (["Advice","Treatment"],["Lời khuyên và phương pháp điều trị {main}","Khi bị {main} nên làm gì và chữa thế nào"])
]

MULTI_TEMPLATES_CTX=[
    (["Symptom","Treatment"],["Triệu chứng và cách điều trị {main} trong {ctx}","{main} khi {ctx}: dấu hiệu và cách chữa"]),
    (["Symptom","Cause"],["Nguyên nhân và triệu chứng của {main} khi {ctx}","Trong {ctx}, {main} có nguyên nhân và dấu hiệu gì"]),
    (["Symptom","Treatment","Cause"],["{main} trong {ctx}: nguyên nhân, triệu chứng và điều trị"]),
    (["Definition","Population"],["Định nghĩa {main} và đối tượng dễ mắc trong {ctx}"]),
    (["Treatment","Complication"],["Biến chứng và cách điều trị {main} khi {ctx}"]),
    (["Cause","Population"],["Nguyên nhân và ai dễ mắc {main} trong {ctx}"]),
    (["RiskFactor","Prevention"],["Yếu tố nguy cơ và cách phòng tránh {main} khi {ctx}"]),
    (["Advice","Treatment"],["Lời khuyên và phương pháp điều trị {main} trong {ctx}"])
]

DISEASE_INTENT=["{x} là bệnh gì","{x} thuộc bệnh nào","{x} là biểu hiện của bệnh gì",
    "{x} có phải bệnh lý không","Nếu bị {x} thì nên đi khám bệnh gì","{x} thuộc nhóm bệnh nào"]

# alias map để map layer graph -> key trong TEMPLATES
LAYER_KEY_MAP = {k:k for k in ALL_FIELDS}
ALIAS_MAP = {
    "disease":"Disease", "subdisease":"Subdisease", "sub_disease":"Subdisease",
    "symptom":"Symptom", "treatment":"Treatment", "cause":"Cause",
    "definition":"Definition", "population":"Population", "advice":"Advice",
    "prevention":"Prevention", "riskfactor":"RiskFactor", "application":"Application",
    "detail":"Detail", "complication":"Complication", "topic":"Topic", "subtopic":"Subdisease"  # subtopic -> Subdisease only if you want
}
LAYER_KEY_MAP.update(ALIAS_MAP)

# ---------- Helpers: generate questions from templates ----------
def _pick_template_for_key(key: str, main: str, ctx: Optional[str]=None, pick_all: bool=False):
    out = []
    if key in TEMPLATES:
        tpls = TEMPLATES[key]
        choices = tpls if pick_all else [random.choice(tpls)]
        for tpl in choices:
            if "{main}" in tpl:
                q = tpl.format(main=main)
            elif "{x}" in tpl:
                q = tpl.format(x=main)
            else:
                q = tpl.replace("{main}", main).replace("{x}", main)
            out.append(q)
    if ctx and key in TEMPLATES_CTX:
        tpls = TEMPLATES_CTX[key]
        choices = tpls if pick_all else [random.choice(tpls)]
        for tpl in choices:
            q = tpl.replace("{main}", main).replace("{ctx}", ctx).replace("{x}", main)
            out.append(q)
    return out

def generate_questions(entity: str, layers: List[str], ctx: Optional[str]=None, pick_all: bool=False, use_multi_templates: bool=True):
    questions = []
    norm_layers = [LAYER_KEY_MAP.get(l, l) for l in layers]
    # single-layer
    for k in norm_layers:
        qs = _pick_template_for_key(k, entity, ctx=ctx, pick_all=pick_all)
        for q in qs:
            questions.append({"layers":[k], "question":q, "template_type":"single"})
    # multi-layer
    if use_multi_templates:
        for labs,tpls in MULTI_TEMPLATES:
            norm = [LAYER_KEY_MAP.get(x, x) for x in labs]
            if all(x in norm_layers for x in norm):
                choices = tpls if pick_all else [random.choice(tpls)]
                for tpl in choices:
                    q = tpl.replace("{main}", entity).replace("{x}", entity)
                    questions.append({"layers":norm, "question":q, "template_type":"multi"})
        for labs,tpls in MULTI_TEMPLATES_CTX:
            norm = [LAYER_KEY_MAP.get(x, x) for x in labs]
            if ctx and all(x in norm_layers for x in norm):
                choices = tpls if pick_all else [random.choice(tpls)]
                for tpl in choices:
                    q = tpl.replace("{main}", entity).replace("{ctx}", ctx).replace("{x}", entity)
                    questions.append({"layers":norm, "question":q, "template_type":"multi_ctx"})
    # dedup + cap
    seen=set(); out=[]
    for it in questions:
        if it["question"] not in seen:
            seen.add(it["question"]); out.append(it)
        if len(out)>=MAX_Q_PER_ENTITY: break
    return out

# ---------- Các hàm support retrieval / extraction ----------
_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!\n])\s+')
def split_sentences(text: str):
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text.strip()]

def token_match_score(entity: str, text: str):
    ent = entity.lower().strip()
    text_l = text.lower()
    if not ent:
        return 0.0
    # exact phrase strong signal
    if ent in text_l:
        return 1.0
    # fallback token overlap (bình thường)
    e_toks = [t for t in re.findall(r'\w+', ent) if len(t) > 1]  # bỏ token 1-char
    if not e_toks:
        return 0.0
    hits = sum(1 for t in set(e_toks) if t and (f" {t} " in f" {text_l} " or t in text_l))
    return hits / len(set(e_toks))

def load_passages(path: Path):
    ps=[]
    with open(path, "r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            j=json.loads(line)
            if "id" not in j: j["id"]=str(i)
            ps.append(j)
    logging.info("Đã load %d passages", len(ps))
    return ps

def load_graph_info(path: Path):
    layer_dict = {}
    # match { name: "....", layer: Layer } or {name: ... , layer: ...}
    pattern = re.compile(r'\{\s*name:\s*["\']?(.*?)["\']?\s*,\s*layer:\s*["\']?(.*?)["\']?\s*\}', re.UNICODE)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for name, layer in pattern.findall(line):
                n = name.strip()
                l = layer.strip()
                if n and l:
                    layer_dict.setdefault(l, set()).add(n)
    logging.info("Đã load graph: %d entity trên %d layer", sum(len(v) for v in layer_dict.values()), len(layer_dict))
    return layer_dict

def find_passages_for_entity(entity_name: str, passages: list, top_k=TOP_K_PASSAGES):
    """
    - Trả về list các passages (ưu tiên những passage chứa exact phrase entity).
    - Nếu passage meta có 'hasanswer' True (nếu bạn đã tiền xử lý), ưu tiên và include.
    """
    ent_low = entity_name.lower().strip()
    scored = []

    # 1) first pass: exact phrase or high token match
    for p in passages:
        text = p.get("text","")
        score = token_match_score(entity_name, text)
        # boost if meta indicates is_selected/hasanswer
        meta = p.get("meta", {}) or {}
        if meta.get("is_selected") or meta.get("hasanswer") or p.get("hasanswer"):
            score += 0.5
        if score > 0:
            scored.append((score,p))

    # 2) if nothing found, do a looser token search
    if not scored:
        for p in passages:
            text = p.get("text","")
            e_toks = [t for t in re.findall(r'\w+', ent_low) if len(t)>1]
            hits = sum(1 for t in e_toks if t in text.lower())
            if hits:
                scored.append((hits/len(e_toks), p))

    # sort and take top_k but also include additional passages that have explicit hasanswer True
    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    # collect top_k plus all passages flagged hasanswer in meta (if any)
    results = []
    seen_ids = set()
    # first top_k
    for score, p in scored[:top_k]:
        pid = p.get("id","")
        results.append({"id":pid, "title":p.get("meta",{}).get("title",""), "text":p.get("text",""), "score":float(score), "hasanswer":True, "isgold": True})
        seen_ids.add(pid)

    # then include others that explicitly mark hasanswer/is_selected
    for score,p in scored:
        pid = p.get("id","")
        meta = p.get("meta",{}) or {}
        if pid in seen_ids: continue
        if meta.get("is_selected") or meta.get("hasanswer") or p.get("hasanswer"):
            results.append({"id":pid, "title":meta.get("title",""), "text":p.get("text",""), "score":float(score), "hasanswer":True, "isgold": True})
            seen_ids.add(pid)

    # finally, if still empty, fallback to top_k of scored
    if not results and scored:
        for score,p in scored[:top_k]:
            pid = p.get("id","")
            results.append({"id":pid, "title":p.get("meta",{}).get("title",""), "text":p.get("text",""), "score":float(score), "hasanswer":True, "isgold": True})

    return results

def extract_answer_sentences_vi(entity: str, ctxs: list, max_sentences=MAX_SENTENCES_PER_ANSWER, neighbor_window=2):
    """
    - Tìm các câu chứa token entity.
    - Khi tìm được câu match, thêm cả neighbor_window câu trước/sau của same passage.
    - Trả về list (sentence, passage_id) sắp xếp theo score (số token trùng).
    """
    ent_toks = set(re.findall(r'\w+', entity.lower()))
    candidates = []  # tuples (score, sent, pid)
    for p in ctxs:
        p_text = p.get("text","")
        sents = split_sentences(p_text)
        for idx, s in enumerate(sents):
            s_low = s.lower()
            # require at least one meaningful token match (ignore very short tokens)
            if any(tok in s_low for tok in ent_toks if len(tok) > 1):
                score = len(set(re.findall(r'\w+', s_low)) & ent_toks)
                # add current sentence
                candidates.append((score, s.strip(), p["id"], idx, p_text))
                # also add neighbors
                for w in range(1, neighbor_window+1):
                    if idx - w >= 0:
                        candidates.append((max(0, score-0.5), sents[idx-w].strip(), p["id"], idx-w, p_text))
                    if idx + w < len(sents):
                        candidates.append((max(0, score-0.5), sents[idx+w].strip(), p["id"], idx+w, p_text))

    # sort by score and de-duplicate while preserving order of highest score first
    candidates = sorted(candidates, key=lambda x: (-x[0]))
    seen = set(); results = []
    for score, sent, pid, sidx, ptext in candidates:
        if sent and sent not in seen:
            results.append((sent, pid))
            seen.add(sent)
        if len(results) >= max_sentences:
            break

    # If still too few and ctxs exist, fallback: include first sentences of ctxs
    if len(results) < max_sentences:
        for p in ctxs:
            for s in split_sentences(p["text"]):
                if s.strip() not in seen:
                    results.append((s.strip(), p["id"]))
                    seen.add(s.strip())
                if len(results) >= max_sentences:
                    break
            if len(results) >= max_sentences:
                break

    return results


# ---------- LLM wrappers ----------
def _safe_parse_json(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}")+1
        return json.loads(text[start:end])
    except Exception as e:
        logging.debug("Không parse được JSON từ LLM output. Exception: %s Output was: %s", e, text[:1000])
        return None

def llm_generate(prompt: str, temperature=LLM_TEMPERATURE):
    # priority: OpenRouter client -> local_generator -> None
    if client:
        try:
            resp = client.chat.completions.create(model=MODEL_ID, messages=[{"role":"user","content":prompt}], temperature=temperature)
            return resp.choices[0].message.content
        except Exception as e:
            logging.warning("OpenRouter call failed: %s", e)
            return None
    elif local_generator:
        try:
            out = local_generator(prompt, max_new_tokens=256, do_sample=False)
            return out[0]["generated_text"]
        except Exception as e:
            logging.warning("Local generator failed: %s", e)
            return None
    else:
        return None

def refine_answer_with_llm_vi(extracted_sentences: list, entity: str, layer: str):
    bullets = "\n".join([f"- {s}" for s in extracted_sentences])
    prompt = textwrap.dedent(f"""
    Dưới đây là các câu trích xuất từ các đoạn văn liên quan đến "{entity}".
    Nhiệm vụ:
    1) Tổng hợp một câu trả lời đầy đủ, rõ ràng và dễ hiểu cho câu hỏi liên quan đến "{entity}" (liên quan tới layer: {layer}).
    2) Nếu có nhiều điểm quan trọng (ví dụ: các bước, phương pháp, loại thuốc, thời điểm...), hãy liệt kê thành từng dòng đầu dòng ("- ...") dưới phần "Điểm chính:".
    3) Chỉ sử dụng thông tin có trong các câu dưới đây. KHÔNG suy đoán hay thêm thông tin bên ngoài.
    4) Nếu không đủ thông tin để trả lời, trả về chuỗi rỗng.
    5) Độ dài: khoảng 1-3 câu tóm tắt + (nếu cần) 3-8 dòng gạch đầu dòng chi tiết. Tổng không quá 1000 ký tự.

    Dữ liệu (các câu trích xuất):
    {bullets}

    Output JSON:
    {{"answer":"..."}}
    Ví dụ (không phải dựa trên dữ liệu trên, chỉ minh hoạ):
    - Input bullets: "- Kem dưỡng ẩm ...\\n- Steroid bôi tại chỗ ...\\n- Kháng histamine uống ... "
    - Output JSON: {{"answer":"Các phương pháp điều trị bao gồm: 1) Kem dưỡng ẩm để giảm khô; 2) Steroid tại chỗ cho trường hợp nặng. Điểm chính:\\n- Kem dưỡng ẩm hoặc thuốc mỡ cho các trường hợp nhẹ.\\n- Steroid bôi tại chỗ cho các trường hợp nặng.\\n- Kháng histamine uống có thể giúp giảm ngứa."}}

    """)
    out = llm_generate(prompt)
    if not out:
        return None
    parsed = _safe_parse_json(out)
    if parsed and "answer" in parsed:
        return parsed["answer"].strip()
    else:
        # fallback: join extracted sentences with separators
        return " ".join(extracted_sentences)[:1000]

def rephrase_question_vi(base_question: str):
    prompt = textwrap.dedent(f"""
    Viết lại câu hỏi sau bằng Tiếng Việt tự nhiên, ngắn gọn, thân thiện:
    Original: {base_question}
    Output JSON:
    {{"question":"..."}}
    """)
    out = llm_generate(prompt, temperature=0.15)
    if not out: return base_question
    parsed = _safe_parse_json(out)
    if parsed and "question" in parsed:
        return parsed["question"].strip()
    return base_question

# ---------- MAIN pipeline ----------
def synthesize_dataset_vi(passages_path=PASSAGES_PATH, graph_path=GRAPH_INFO_PATH, out_path=OUTPUT_PATH):
    passages = load_passages(passages_path)
    layer_dict = load_graph_info(graph_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_q = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for layer, entities in layer_dict.items():
            for entity in entities:
                layers_for_entity = [layer]
                ctx = None
                q_items = generate_questions(entity, layers_for_entity, ctx=ctx, pick_all=False)
                if not q_items:
                    base = generate_question_from_template_vn(entity, layer)
                    q_items = [{"layers":[layer], "question": base, "template_type":"fallback"}]

                for qi in q_items:
                    question_text = qi["question"]

                    ctxs = find_passages_for_entity(entity, passages, top_k=TOP_K_PASSAGES)
                    if not ctxs:
                        # nothing to answer from passages -> skip
                        continue

                    pool = [p for p in passages if p.get("id") not in {c["id"] for c in ctxs}]
                    negs=[]
                    if pool:
                        sample_negs = random.sample(pool, min(NEGATIVE_CTXS, len(pool)))
                        for n in sample_negs:
                            negs.append({"id":n.get("id",""), "title": n.get("meta",{}).get("title",""), "text": n.get("text",""), "hasanswer":False, "isgold":False})
                    final_ctxs = ctxs + negs

                    extracted = extract_answer_sentences_vi(entity, ctxs, max_sentences=MAX_SENTENCES_PER_ANSWER)
                    sentences = [s for s,_ in extracted]

                    answer_text = ""
                    if sentences:
                        if REFINE_ANSWER_WITH_LLM and (client or local_generator):
                            ans = refine_answer_with_llm_vi(sentences, entity, layer)
                            answer_text = ans if ans is not None else " ".join(sentences)
                        else:
                            answer_text = " ".join(sentences)
                    else:
                        combined = " ".join([c["text"] for c in ctxs])
                        if combined and (client or local_generator):
                            prompt = textwrap.dedent(f"""
                            Tổng hợp các đoạn sau về '{entity}' và trả lời đầy đủ. Chỉ dùng thông tin có sẵn.
                            Đoạn:
                            {combined}
                            Output JSON: {{"answer":"..."}}
                            """)
                            raw = llm_generate(prompt)
                            parsed = _safe_parse_json(raw) if raw else None
                            if parsed and "answer" in parsed:
                                answer_text = parsed["answer"].strip()
                            else:
                                answer_text = combined[:1000]
                        else:
                            answer_text = ""

                    q_out = question_text
                    if (client or local_generator):
                        q_reph = rephrase_question_vi(question_text)
                        if q_reph: q_out = q_reph

                    example = {"question": q_out, "answers": [answer_text] if answer_text else [], "ctxs": final_ctxs}
                    fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                    total_q += 1
    logging.info("Hoàn tất. Lưu dataset tại: %s  (tổng câu hỏi: %d)", out_path, total_q)

# fallback helper
def generate_question_from_template_vn(entity: str, layer: str):
    key = LAYER_KEY_MAP.get(layer, layer)
    if key in TEMPLATES:
        return random.choice(TEMPLATES[key]).replace("{main}", entity).replace("{x}", entity)
    if key in TEMPLATES_CTX:
        return random.choice(TEMPLATES_CTX[key]).replace("{main}", entity).replace("{ctx}", "")
    return f"{entity} là gì?"

# ---------- chạy ----------
if __name__ == "__main__":
    synthesize_dataset_vi()
