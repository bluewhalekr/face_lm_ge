import json

data = {
    "파일경로": "/사용자/문서/프로젝트/파일.txt",
    "설명": "이 경로는 한글로 작성되었습니다."
}

# JSON 파일로 저장
with open('./data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
