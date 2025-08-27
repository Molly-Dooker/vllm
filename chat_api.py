import json
import copy
import requests

API_URL = "http://127.0.0.1:7000/v1/chat/completions"  # 호스트:7000 → 컨테이너:8000

def load_template(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ask_from_template(template: dict, content: str) -> str:
    payload = copy.deepcopy(template)  # 중첩객체 보호
    assert "messages" in payload and isinstance(payload["messages"], list), "template에 messages가 없습니다."

    # 규칙: 마지막 user 메시지를 찾아 덮어쓰기 (없으면 append)
    user_idx = None
    for i in reversed(range(len(payload["messages"]))):
        if payload["messages"][i].get("role") == "user":
            user_idx = i
            break

    if user_idx is None:
        payload["messages"].append({"role": "user", "content": content})
    else:
        payload["messages"][user_idx]["content"] = content

    r = requests.post(API_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# 사용 예시
TEMPLATE = load_template("chat_template.json")
response = ask_from_template(TEMPLATE, "사무실에 책상이 있어?")
