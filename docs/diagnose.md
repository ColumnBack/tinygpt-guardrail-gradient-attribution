# 종합 진단 (`diagnose.py`)

오판 하나에 네 기법을 모두 적용하고, 마지막으로 학습 데이터를 본다.
기법 하나는 가설이고, **여러 기법이 같은 토큰과 같은 경로를 가리키면 근거**가 된다. 어긋나면 그것도 발견이다.

| 단계 | 기법 | 질문 |
|---|---|---|
| 1 | IG ($S = Z[\text{<call>}, \text{오답}]$) | 어떤 토큰이 오답을 끌어올렸나 |
| 2 | occlusion (zero) | 어떤 토큰을 지우면 정답으로 돌아오나 |
| 3 | 층별 hidden state 비교 | 몇 번째 층부터 `<call>`이 계속 오답 쪽인가 |
| 4 | attention (그 층) | `<call>`이 그 토큰에서 정보를 가져오나 |
| 5 | 학습 데이터 | 모델이 왜 그 연결을 배웠나 |

## 예: `draft` 사례

```
A: the server shows an error so help me read the log file  -> filesystem.read_file  (correct)
B: the draft  shows an error so help me read the log file  -> filesystem.write_file (wrong)

 1. IG           top: draft 20.8, file 4.8, help 0.8          -> 'draft'  ✓
 2. occlusion    top drop: draft 21.1, log 2.2, read 1.2       removing 'draft' -> read_file  ✓
 3. hidden state <call> stays on the wrong tool from L2.attn
 4. attention    layer 2:  A takes most from 'log' 120.6,  B from 'draft' 115.7  ✓
 5. data         'draft' in training: write_file 17, list_directory 1, web.fetch 1

 VERDICT: 'draft' caused the wrong decision; it reaches <call> through the layer-2 attention.
```

결론을 한 문장으로 쓰면: **`draft`가 2층 attention을 통해 결정 위치로 들어가 도구 선택을 뒤집었고,
그 연결은 학습 데이터에서 `draft`가 거의 항상 파일 쓰기와 함께 나왔기 때문에 생겼다.**

(IG 값은 적분 64단계 기준이라 256단계로 계산한 다른 문서의 값과 소수점이 조금 다르다.)

## 모든 오판 사례 (`--all`)

| 바뀐 단어를 가리키나 | 처음 모델 (207건) | 보강 후 (69건) |
|---|---|---|
| IG 1위 | 88% | 71% |
| occlusion 1위 | 88% | 93% |
| 지우면 정답으로 돌아옴 | 100% | 100% |
| attention 1위 (오답으로 넘어간 층) | 66% | 58% |
| **네 기법 모두 일치** | **60%** | **39%** |

- 처음 모델은 오판의 60%를 네 기법이 한목소리로 설명한다.
- **일치하지 않는 경우가 공부거리다.** attention이 1위로 꼽지 않는 경우가 가장 많다.
  가능한 이유(아직 확인하지 않은 가설): 정보가 attention 한 번이 아니라 여러 위치를 거쳐 들어오거나
  (1층에서 다른 토큰으로 옮겨진 뒤 2층에서 `<call>`로), FFN에서 증폭되는 경우다.
  층별 hidden state 비교의 위치별 지도로 그 경로를 따라가 확인할 수 있다.
- 보강 후 모델에서는 일치율이 39%로 낮다. 남은 오판(`weekend`, `folder` 등)이 단순한 "단어 하나 = 도구" 연결이 아니라서
  기법마다 다른 면을 보고 있을 가능성이 있다(확인 필요).

## 실행

```bash
python diagnose.py                  # draft 사례
python diagnose.py --index 5        # audit가 찾은 5번째 오판
python diagnose.py --all            # 모든 오판에서 기법 일치율
python diagnose.py --augment --all
```
