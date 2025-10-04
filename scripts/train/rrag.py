import pandas as pd
import math, hashlib, unicodedata, random, argparse
from typing import List, Sequence, Tuple, Dict
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# sklearn metrics
_HAS_SK=True
try:
    from sklearn.metrics import average_precision_score,f1_score
except Exception:
    _HAS_SK=False
# -------------------- Labels --------------------
ALL_FIELDS=[
    "disease","subdisease","topic","subtopic",
    "complication","treatment","application","cause",
    "definition","detail","population","advice",
    "riskfactor","prevention","symptom"
]
IDX={k:i for i,k in enumerate(ALL_FIELDS)}
NUM_CLASSES=len(ALL_FIELDS)

# -------------------- Synthetic space --------------------
DISEASES=[
    # base 10
    "viêm âm đạo","loạn khuẩn âm đạo","u xơ tử cung","mãn kinh","tiền mãn kinh",
    "thống kinh","viêm cổ tử cung","đa nang buồng trứng","nhiễm nấm âm đạo","ung thư cổ tử cung",
    # thêm 20
    "lạc nội mạc tử cung","u nang buồng trứng","viêm nội mạc tử cung","viêm buồng trứng",
    "viêm vùng chậu","sẩy thai","thai ngoài tử cung","thai lưu","sinh non",
    "tiểu không kiểm soát","viêm đường tiết niệu","nhiễm HPV","hội chứng buồng trứng đa nang",
    "ung thư buồng trứng","ung thư nội mạc tử cung","rong kinh","vô kinh","đa kinh",
    "đau bụng kinh","viêm niệu đạo"
]

SYMPTOMY=[
  "mệt mỏi","đau khi quan hệ tình dục","ngứa vùng kín","khí hư có mùi hôi",
  "ra máu sau quan hệ","đau vùng chậu","tiểu buốt","tiểu rát","bốc hỏa",
  "mất ngủ","rối loạn kinh nguyệt","trễ kinh nhiều ngày","đau bụng kinh dữ dội",
  "khó chịu khi tiểu","chảy máu âm đạo bất thường","đau lưng dưới",
  "đau bụng dưới","khô âm đạo","đi tiểu nhiều lần","đau rát âm hộ"
]

TOPICS=[
    "Mang thai","Thai kỳ","Ba tháng đầu","Ba tháng cuối","Nghén",
    "Tiểu đường thai kỳ","Đi bộ","Siêu âm","Yoga","Xét nghiệm Pap smear",
    # thêm
    "Sinh non","Cho con bú","Khám phụ khoa","Dậy thì","Nội tiết tố",
    "Xét nghiệm máu","Xét nghiệm nước tiểu","Phẫu thuật","Dinh dưỡng",
    "Tiền sản giật","Huyết áp cao","Thiếu máu","Vaccine HPV","Khám sức khỏe định kỳ"
]
SUBTOPICS=["Tam cá nguyệt đầu","Tam cá nguyệt cuối"]

# -------------------- Templates đơn lớp --------------------
TEMPLATES = {
  "symptom": [
    "Triệu chứng của {main}","Dấu hiệu {main}","Biểu hiện khi mắc {main}",
    "Các dấu hiệu nhận biết {main}","Triệu chứng đặc trưng của {main}",
    "Khi nào thì biết mình đang mắc {main}","Làm sao nhận biết {main}",
    "{main} có dấu hiệu ban đầu ra sao","Có triệu chứng tiềm ẩn nào của {main} không"
  ],
  "cause": [
    "Nguyên nhân gây {main}","Vì sao bị {main}","Điều gì gây ra {main}",
    "Lý do khiến dễ bị {main}","Nguyên nhân sâu xa dẫn tới {main}",
    "Tại sao nhiều người mắc {main}","{main} xuất hiện từ yếu tố nào",
    "Nguyên nhân phổ biến nhất của {main}","Có yếu tố môi trường nào khiến {main} không"
  ],
  "treatment": [
    "Điều trị {main}","Cách chữa {main}","Phác đồ cho {main}",
    "Có thuốc nào dùng cho {main}","Điều trị {main} có cần phẫu thuật không",
    "Chữa {main} tại nhà có hiệu quả không","{main} điều trị bằng thuốc gì",
    "Các phương pháp y học hiện đại chữa {main}","Khi mắc {main} thì bác sĩ thường chỉ định gì"
  ],
  "complication": [
    "Biến chứng của {main}","Hậu quả nếu bị {main}","Rủi ro lâu dài của {main}",
    "{main} nếu không chữa có nguy hiểm không","Bệnh {main} có thể dẫn đến vô sinh/ung thư không",
    "Nguy cơ về sức khỏe khi bị {main}","{main} có gây di chứng gì không",
    "Nếu {main} tiến triển nặng sẽ ra sao","{main} có làm tăng nguy cơ tử vong không"
  ],
  "population": [
    "Ai dễ bị {main}","Đối tượng nào thường mắc {main}","{main} thường gặp ở lứa tuổi nào",
    "{main} phổ biến ở phụ nữ sau sinh không","Nhóm tuổi nào dễ mắc {main}",
    "{main} thường gặp ở nam hay nữ","Ai có khả năng mắc {main} cao nhất",
    "Đối tượng đặc biệt nào cần lưu ý về {main}","{main} có thường gặp ở trẻ em không"
  ],
  "definition": [
    "{main} là gì","Định nghĩa {main}","Khái niệm về {main}",
    "Khái niệm cơ bản về {main}","Tóm tắt {main}","{main} có nghĩa là gì trong y học",
    "Định nghĩa ngắn gọn {main}","{main} là thuật ngữ chỉ điều gì",
    "Bạn hiểu thế nào về {main}"
  ],
  "detail": [
    "Mô tả chi tiết về {main}","Thông tin thêm về {main}","{main}: thông tin quan trọng",
    "Những điều cần biết về {main}","Mô tả kỹ hơn {main}","Chi tiết quan trọng về {main}",
    "Cung cấp thêm thông tin về {main}","{main} bao gồm những gì",
    "Những khía cạnh cần lưu ý về {main}"
  ],
  "subdisease": [
    "Các thể của {main}","Phân loại {main}","{main} gồm các nhóm nào",
    "Các loại {main}","Phân nhóm bệnh {main}","{main} có mấy thể chính",
    "Kể tên các dạng {main}","Bệnh {main} chia thành những thể nào",
    "Những phân nhóm đặc trưng của {main}"
  ],
  "application": [
    "Ứng dụng của {main}","{main} được dùng để làm gì","Vai trò {main} trong y khoa",
    "{main} được dùng để làm gì trong y tế","Vai trò thực tế của {main}",
    "Ý nghĩa lâm sàng của {main}","{main} hỗ trợ bác sĩ trong việc gì",
    "Tác dụng chính của {main}","{main} thường áp dụng trong trường hợp nào"
  ],
  "advice": [
    "Lời khuyên khi bị {main}","Nên làm gì khi có {main}","Khi {main} thì cần lưu ý gì",
    "Cách xử lý an toàn khi gặp {main}","Nếu có {main} thì cần kiêng gì",
    "Bác sĩ thường khuyên gì cho {main}","Khi {main} thì nên làm gì",
    "Khi {main} thì nên đi khám ngay không","Những điều cần tránh khi có {main}"
  ],
  "riskfactor": [
    "Yếu tố nguy cơ của {main}","Ai có nguy cơ cao bị {main}","Tình huống nào dễ dẫn tới {main}",
    "Người nào có nguy cơ cao bị {main}","Yếu tố làm tăng khả năng mắc {main}",
    "Thói quen nào dễ gây {main}","Ai cần đi khám sớm vì nguy cơ {main}",
    "Có bệnh nền nào khiến {main} nặng hơn không","Lối sống nào làm tăng nguy cơ {main}"
  ],
  "prevention": [
    "Cách phòng ngừa {main}","Làm sao để tránh {main}","Phòng bệnh {main} thế nào",
    "Cách tốt nhất để phòng tránh {main}","Làm thế nào để ngừa {main}",
    "Có vaccine hoặc thuốc dự phòng {main} không","Thói quen giúp hạn chế {main}",
    "Chế độ sinh hoạt nào giúp phòng tránh {main}","Thực phẩm nào giúp ngừa {main}"
  ],
  "disease": [
    "{x} là bệnh gì","{x} là bệnh gì?","{x} thuộc bệnh nào",
    "{x} liên quan tới bệnh nào","{x} là biểu hiện của bệnh gì",
    "{x} thường gặp trong bệnh gì","{x} có thể do bệnh gì gây ra",
    "{x} là dấu hiệu của bệnh nào","Nếu bị {x} thì có thể đang mắc bệnh gì"
  ]
}


# -------------------- Templates có context --------------------
TEMPLATES_CTX={
  "symptom":["Triệu chứng của {main} trong {ctx}","Dấu hiệu {main} khi {ctx}"],
  "cause":["Nguyên nhân {main} trong {ctx}","Tại sao có {main} khi {ctx}"],
  "treatment":["Điều trị {main} trong {ctx}","Xử lý {main} khi {ctx}"],
  "complication":["Biến chứng {main} trong {ctx}","Rủi ro của {main} khi {ctx}"],
  "population":["{main} trong {ctx} thường gặp ở ai","Ai hay mắc {main} khi {ctx}"],
  "definition":["Định nghĩa {main} trong {ctx}","{main} là gì khi {ctx}"],
  "detail":["Thông tin chi tiết {main} trong {ctx}","Điều cần biết về {main} khi {ctx}"],
  "subdisease":["Các thể {main} trong {ctx}","Phân loại {main} khi {ctx}"],
  "application":["Ứng dụng của {main} trong {ctx}","{main} được dùng thế nào khi {ctx}"],
  "advice":["Lời khuyên khi bị {main} trong {ctx}","Nên làm gì với {main} khi {ctx}"],
  "riskfactor":["Yếu tố nguy cơ {main} trong {ctx}","Nguy cơ mắc {main} khi {ctx}"],
  "prevention":["Phòng ngừa {main} trong {ctx}","Làm sao tránh {main} khi {ctx}"]
}

# -------------------- Multi-layer templates --------------------
MULTI_TEMPLATES=[
    (["symptom","treatment"],["Triệu chứng và cách điều trị {main}","Dấu hiệu của {main} và chữa thế nào"]),
    (["symptom","cause"],["Triệu chứng và nguyên nhân của {main}","Vì sao mắc {main} và có triệu chứng gì"]),
    (["symptom","treatment","cause"],["Triệu chứng, nguyên nhân và cách chữa {main}","Nguyên nhân, triệu chứng và điều trị {main}"]),
    (["definition","population"],["{main} là gì và ai dễ mắc","Định nghĩa và đối tượng nguy cơ của {main}"]),
    (["treatment","complication"],["Điều trị {main} và các biến chứng có thể gặp","{main}: biến chứng và hướng điều trị"]),
    (["cause","population"],["Nguyên nhân và đối tượng dễ mắc {main}","{main}: vì sao bị và thường gặp ở ai"]),
    (["subdisease","symptom"],["Các thể {main} và triệu chứng đi kèm","Phân loại {main} cùng dấu hiệu nhận biết"]),
    (["riskfactor","prevention"],["Yếu tố nguy cơ và cách phòng tránh {main}","Làm sao để phòng ngừa {main} và ai dễ mắc"]),
    (["advice","treatment"],["Lời khuyên và phương pháp điều trị {main}","Khi bị {main} nên làm gì và chữa thế nào"])
]

# -------------------- Multi-layer + context --------------------
MULTI_TEMPLATES_CTX=[
    (["symptom","treatment"],["Triệu chứng và cách điều trị {main} trong {ctx}","{main} khi {ctx}: dấu hiệu và cách chữa"]),
    (["symptom","cause"],["Nguyên nhân và triệu chứng của {main} khi {ctx}","Trong {ctx}, {main} có nguyên nhân và dấu hiệu gì"]),
    (["symptom","treatment","cause"],["{main} trong {ctx}: nguyên nhân, triệu chứng và điều trị"]),
    (["definition","population"],["Định nghĩa {main} và đối tượng dễ mắc trong {ctx}"]),
    (["treatment","complication"],["Biến chứng và cách điều trị {main} khi {ctx}"]),
    (["cause","population"],["Nguyên nhân và ai dễ mắc {main} trong {ctx}"]),
    (["riskfactor","prevention"],["Yếu tố nguy cơ và cách phòng tránh {main} khi {ctx}"]),
    (["advice","treatment"],["Lời khuyên và phương pháp điều trị {main} trong {ctx}"])
]

# -------------------- Disease intent --------------------
DISEASE_INTENT=["{x} là bệnh gì","{x} thuộc bệnh nào","{x} là biểu hiện của bệnh gì",
    "{x} có phải bệnh lý không","Nếu bị {x} thì nên đi khám bệnh gì","{x} thuộc nhóm bệnh nào"]


# -------------------- Data builder --------------------
def build_train_raw()->List[Tuple[str,List[str],int]]:
    rows=[]

    for k,tpls in TEMPLATES.items():
        if k == "disease":
            # 1. disease intent từ SYMPTOMY
            for x in SYMPTOMY:
                for tpl in tpls:
                    if "{x}" in tpl:
                        rows.append((tpl.format(x=x), [k], 0))
            # 2. disease từ chính DISEASES
            for d in DISEASES:
                for tpl in tpls:
                    if "{main}" in tpl:
                        rows.append((tpl.format(main=d), [k], 0))
        else:
            # các lớp khác dựa trên DISEASES
            for d in DISEASES:
                # single-label (no context)
                for tpl in tpls:
                    rows.append((tpl.format(main=d), [k], 0))

                # single-label + context (nếu có trong TEMPLATES_CTX)
                if k in TEMPLATES_CTX:
                    for ctx in TOPICS+SUBTOPICS:
                        for tpl_ctx in TEMPLATES_CTX[k]:
                            rows.append((tpl_ctx.format(main=d, ctx=ctx), [k], 1))

    # multi-label (no context)
    for d in DISEASES:
        for labs,tpls in MULTI_TEMPLATES:
            for tpl in tpls:
                rows.append((tpl.format(main=d), labs, 0))

        # multi-label + context
        for ctx in TOPICS+SUBTOPICS:
            for labs,tpls in MULTI_TEMPLATES_CTX:
                for tpl in tpls:
                    rows.append((tpl.format(main=d, ctx=ctx), labs, 1))

        # topic/subtopic root (cũng tính là có context)
        for ctx in TOPICS:
            rows.append((f"{d} trong {ctx}", ["topic"], 1))
        for ctx in SUBTOPICS:
            rows.append((f"{d} ở {ctx}", ["subtopic"], 1))

    # dedup
    seen=set(); out=[]
    for text,labs,has_ctx in rows:
        key=(text.lower(),"|".join(sorted(labs)),has_ctx)
        if key not in seen:
            seen.add(key)
            out.append((text,labs,has_ctx))
    return out


