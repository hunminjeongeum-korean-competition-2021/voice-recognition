# 음성인식 성능평가 인공지능 경진대회

- [문제3 음성 인식(명령어) 데이터셋 설명](#문제-3-음성인식-명령어-dataset-설명)
- [문제4 음성 인식(자유대화) 데이터 셋명](#문제-4-음성인식-자유대화-dataset-설명)

## 대회 규칙

- **주제**
  - 한국어 음성을 텍스트로 변환하는 인공지능 개발
- **평가**
  - CER(Character Error Rate) 제1지표
  - WER(Word Error Rate) 제2지표
  - 리더보드에는 CER만 표시
  - 최종평가시 동점인 경우 차순위 지표 우수팀이 상위
  - 최종 동점이 발생한 경우 먼저 제출한 팀이 상위
- **NSML GPU 지원**
  - Tesla V100-SXM2-32GB 2개 (하나의 task에 2개 사용 가능)
- **<u>법적 제약이 없는 외부 데이터 및 사전 학습 모델 사용 가능(대회 종료 후 코드 제출시 함께 제출)</u>**

## 문제 3 **[음성인식-명령어 Dataset 설명]**

- 음성 파일과 매칭되는 Text를 추론(aka.받아쓰기)

  | 전체 크기 |                  파일수                  |  NSML 데이터셋 이름 |
  | :-------: | :--------------------------------------: | :-----: |
  |   33GB    | train_data(113,347)<br>test_data(10,374) | stt_1 |

### Train Dataset

- `root_path/train/train_data/`(113,347개의 wav 파일 \*확장자 없는 레이블 형태)

  ```
  idx000001
  idx000002
  idx000003
  idx000004
  ...
  idx113344
  idx113345
  idx113346
  idx113347
  ```

### Train Label

- `root_path/train/train_label`

- `train_label (DataFrame 형식, 113,347rows)`

  - columns - `["file_name", "text"]`

  - `file_name` - train_data 폴더에 존재하는 wav파일명 (ex. idx000001)

  - `text` - train_data 폴더에 존재하는 wav파일과 매칭되는 text 정보 (ex. 훈민정음에 스며들다)

### Test Data

- `root_path/test/test_data/`(10,374개의 wav 파일 \*확장자 없는 레이블 형태 / train_data와 파일명 형식이 다름에 주의)

  ```
  idx_000001
  idx_000002
  idx_000003
  idx_000004
  ...
  idx_010371
  idx_010372
  idx_010373
  idx_010374
  ```

### Test Label

- `root_path/test/test_label` (참가자 접근 불가)

- `test_label (DataFrame 형식, 10,374rows)`

- columns = `["file_name", "text"]`

  - `file_name`- test_data 폴더에 존재하는 wav파일명 (ex. idx_000001)

  - `text` - test_data 폴더에 존재하는 wav파일과 매칭되는 Text 정보 (ex. 훈민정음에 스며들다)

  - 참가자 분들은 `file_name`과 `text` column을 모두 기입한 DataFrame을 결과물로 구성 (최종 제출 format)

---

## 문제 4 **[음성인식-자유대화 Dataset 설명]**

- 음성 파일과 매칭되는 text를 추론

  | 전체 크기 |                 파일수                  | NSML 데이터셋 이름 |
  | :-------: | :-------------------------------------: | :----: |
  |  35.44GB  | train_data(228,913)<br>test_data(9,436) | stt_2 |

### Train Dataset

- `root_path/train/train_data/`(228,913개의 wav 파일 \*확장자 없는 레이블 형태)

  ```
  idx000001
  idx000002
  idx000003
  idx000004
  ...
  idx228910
  idx228911
  idx228912
  idx228913
  ```

### Train Label

`root_path/train/train_label`

- `train_label (DataFrame 형식, 228,913rows)`

  - columns - `["file_name", "text"]`

  - `file_name` - train_data 폴더에 존재하는 wav파일명 (ex. idx000001)

  - `text` - train_data 폴더에 존재하는 wav파일과 매칭되는 Text 정보 (ex. 훈민정음에 스며들다)

### Test Dataset

- `root_path/test/test_data/`(9,436개의 wav 파일 \*확장자 없는 레이블 형태/ train_data와 파일명 형식이 다름에 주의)

  ```
  idx_000001
  idx_000002
  idx_000003
  idx_000004
  ...
  idx_009433
  idx_009434
  idx_009435
  idx_009436
  ```

### Test Label

- `root_path/test/test_label` (참가자 접근 불가)

- `test_label (DataFrame 형식, 9,436rows)`

  - columns = `["file_name", "text"]`

  - `file_name` - test_data 폴더에 존재하는 wav파일명 (ex. idx_000001)

  - `text` - test_data 폴더에 존재하는 wav파일과 매칭되는 Text 정보 (ex. 훈민정음에 스며들다)

  - 참가자 분들은 `file_name`과 `text` column을 모두 기입한 DataFrame을 결과물로 구성(최종 제출 format)
