# 대화 요약 인공지능 경진대회

- [문제1 문서 요약 데이터셋 설명](#문제-1-문서요약-dataset-설명)

## 대회 규칙

- **주제**
  - 한국어 원문으로부터 생성요약문을 도출해내는 인공지능 개발
- **평가**
  - Rouge-L(제1지표)
  - Rouge-2(제2지표)
  - Rouge-1(제3지표)
  - 리더보드에는 Rouge-L만 적용
  - 최종평가시 동점인 경우 차순위 지표 우수팀이 상위
  - 최종 동점이 발생한 경우 먼저 제출한 팀이 상위
- **NSML GPU 지원**
  - Tesla V100-SXM2-32GB 1개
- **<u>외부 데이터 및 사전 학습 모델 사용 불가</u>**

### 문제 1 **[문서요약 Dataset 설명]**

- 텍스트로 구성된 대화문을 읽어 요약문 추론

  | 전체 크기 |            파일수             | NSML 데이터셋 이름 |
  | :-------: | :---------------------------: | :--------:|
  |   436MB   | train_data(5)<br>test_data(5) | dialogue |

#### Train Dataset

- `root_path/train/train_data/`(Edu, Food, Life, Per_Rel, Work)
- train_label은 따로 존재하지 않으며 train_data의 summary를 추출하여 label로 사용

  ```
  train_data (json): 학습용 데이터셋
  ├ numberOfItems (ex. 71408)
  ├ data
  │├ header
  ││├ dialogueInfo
  │││├ numberOfParticipants (ex. 2)
  │││├ numberOfUtterances (ex. 16)
  │││├ numberOfTurns (ex. 6)
  │││├ type (ex. 일상 대화)
  │││├ topic (ex. 개인 및 관계)
  │││├ dialogueID (ex. b6d7d466-25f2-5b3e-a55d-fe5ce1e32687)
  ││├ participantsInfo (List)
  │││├ age (ex. 20대)
  │││├ residentialProvince (ex. 부산광역시)
  │││├ gender (ex. 여성)
  │││├ participantID (P01)
  +
  │││├ ...
  │├ body
  ││├ dialogue (List)
  │││├ utterance (ex. 양치하고 올께)
  │││├ utteranceID (ex. U1)
  │││├ participantID (ex. P01)
  │││├ date (ex. 2019-11-08)
  │││├ turnID (ex. T1)
  │││├ time (ex. 13:32:00)
  +
  │││├ ...
  ││├ summary (ex. 양치를 하고 온다고 하며 커피가 치아에 좋지 않다는 이야기를 한다.)
  ```

#### Test Dataset

- `root_path/test/test_data/`(Edu, Food, Life, Per_Rel, Work)
- 5개의 json 파일로 이루어진 test_data의 summary를 추론하는 문제

  ```
  test_data (json): 추론용 데이터셋
  ├ numberOfItems (ex. 71408)
  ├ data
  │├ header
  ││├ dialogueInfo
  │││├ numberOfParticipants (ex. 2)
  │││├ numberOfUtterances (ex. 16)
  │││├ numberOfTurns (ex. 6)
  │││├ type (ex. 일상 대화)
  │││├ topic (ex. 개인 및 관계)
  │││├ dialogueID (ex. b6d7d466-25f2-5b3e-a55d-fe5ce1e32687)
  ││├ participantsInfo (List)
  │││├ age (ex. 20대)
  │││├ residentialProvince (ex. 부산광역시)
  │││├ gender (ex. 여성)
  │││├ participantID (P01)
  +
  │││├ ...
  │├ body
  ││├ dialogue (List)
  │││├ utterance (ex. 양치하고 올께)
  │││├ utteranceID (ex. U1)
  │││├ participantID (ex. P01)
  │││├ date (ex. 2019-11-08)
  │││├ turnID (ex. T1)
  │││├ time (ex. 13:32:00)
  +
  │││├ ...
  ││├ summary (ex. 추론 해야하는 요약문)
  ```

#### Test Label

- test_label (DataFrame 형식, 참가자 접근 불가)

- columns - `["id", "summary"]`

  - `id` - dialogueID와 동일

  - `summary` - 추론한 요약문을 기입하여 제출

  - `id`와 `summary` column을 모두 기입한 DataFrame을 결과물로 구성(**최종 제출 format**)
