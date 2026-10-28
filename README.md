# sdi2026-galaxy-sfr

제1회 과학데이터혁신 경진대회(서울대 IDIS DataHub, 2026) — **은하별 별 생성률(SFR) 예측** 팀 저장소.

## 일정
| 날짜 | 할 일 |
|---|---|
| 2026-11-04 (수) 12:00 | 중간 보고서: 템플릿 Part I (1~8절) |
| 2026-11-13 (금) | 최종 보고서 + 코드: 같은 노트북 Part II (9~16절) |
| 2026-11-25 (수) 18:00 | 상위 팀 대면 발표 (진출 시) |

## 파일
- `DataHub_Midterm_Final_Report_Template.ipynb`: 대회에서 받은 보고서 템플릿 원본. 제출용 노트북은 이걸 복사해 만든다.
  - `# DO NOT MODIFY` 칸(임포트·시드, 스키마 확인, 공통 분할, 지표, 기준선, 최종 평가)은 고치지 않는다.
  - 제출 파일 이름: `DataHub_<ApplicationID>_mid.ipynb`, `DataHub_<ApplicationID>_final.ipynb`.
- `Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv`: 데이터(45MB). 노트북이 이 이름 그대로 같은 폴더에서 찾으니 이름을 바꾸지 않는다.

## 역할 분담과 일정 (중간 보고서, 2026-10-28 정함)
| 누구 | 맡는 것 | 템플릿 절 | 작업 폴더 |
|---|---|---|---|
| 조장 (ZizonK7) | **머신러닝**: 검증 전략, 기준선 해석, 예비 모델(손실 비교·2단계 모델) | 5.1, 6.1, 7·7.1 | `work/조장/` |
| 팀원 (james092-source) | **EDA**: 데이터 점검, 탐색 그림(모델 결정에 영향을 준 그림 2개 이상)과 해석, 피처 설명 | 2.1, 3·3.1, 4.1 | `work/팀원/` |
| 같이 | **문제 정의**, **요약·다음 계획** | 1, 8 | 제출 노트북 |

**일정**
1. **~11-02 (월)**: 둘 다 `study/` 수업정리 **4개 모두** 공부한다(1-1 → 1-2 → 2-1 → 2-2). 그리고 각자 맡은 절 작업을 자기 폴더에서 한다.
2. **11-02 (월) 만남**:
   1. 서로 맡은 부분을 간략히 보고한다(그림·숫자가 무슨 뜻인지, 무엇을 결정했는지).
   2. 조장이 두 작업을 제출용 노트북 하나로 합친다.
   3. 같이 1절(문제 정의)과 8절(요약·다음 계획)을 쓴다.
3. **11-04 (수) 12:00**: 중간 보고서 제출. 같은 날 11:00에 조장 시험이 있어서 월요일 밤에 끝내는 것이 목표.

- **EDA와 모델이 끊기지 않게**: 팀원은 EDA 결과마다 "그래서 모델에서 무엇을 바꿔야 하나"를 한 줄씩 붙인다(예: 0 비율이 주변 밀도에 따라 크게 다르다 → 0인가 분류에 밀도가 중요). 조장은 그 줄을 읽고 7절 실험을 고른다.

## 같이 작업하는 방법 (git 명령 없이)
1. **각자 자기 노트북에서 작업한다.** 예: `work/이름_eda.ipynb`. 제출용 노트북은 한 사람만 고친다. 노트북은 두 사람이 같은 파일을 고치면 합치기가 어렵다.
2. **올리기는 웹에서**: 레포 페이지 → Add file → Upload files → 파일 끌어 놓기 → 아래에서 **"Create a new branch for this commit and start a pull request"** 선택 → Propose changes → Create pull request.
3. main에는 PR로만 합친다. 직접 main에 올리지 않는다.

## 규칙 (설명회 기준)
- 제공된 데이터만 쓴다(추가 데이터 금지).
- AI 사용 자유. 다른 사람·다른 곳에 올라온 코드를 그대로 베끼는 건 금지.
- 대회 중에는 저장소를 비공개로 둔다.

## 데이터 출처
"Galaxy Features with Position and Velocity for Star Formation Rate Prediction", Keita Fukushima & Prof. Ji-hoon Kim (Department of Physics & Astronomy, SNU), derived from the Crocodile simulation project (https://sites.google.com/view/crocodilesimulation/home), via SNU IDIS DataHub. Licensed under **CC BY 4.0**.
