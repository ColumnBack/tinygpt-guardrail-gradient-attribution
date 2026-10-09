# Attention 분석

IG가 **어떤** 토큰이 결정을 만들었는지, 층별 hidden state 비교가 **몇 번째 층에서** 갈라지는지 알려 준다면,
attention 분석은 결정 위치 `<call>`이 **어디에서 정보를 가져오는지** 보여 준다.
코드는 `attention_analysis.py`이고, `tinygpt.py`는 읽기만 한다.

## 무엇을 보나

블록 n, 헤드 h의 attention 가중치는 causal mask를 쓴 softmax다.

$$
A^{(n,h)} = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}} + \text{mask}\right), \qquad k = \text{<call> 위치}
$$

같은 k행을 세 가지로 본다.

| 보기 | 식 | 뜻 |
|---|---|---|
| weight | $A^{(n,h)}[k, j]$ | `<call>`이 토큰 j에 주는 가중치 (헤드 평균도 출력) |
| norm | $\left\lVert \sum_h A^{(n,h)}[k,j]\; V^{(n,h)}[j]\, W_O^{(n,h)} \right\rVert$ | 토큰 j가 `<call>`의 attention 출력에 **실제로 더하는 벡터의 크기** |
| rollout | $R = \tilde A_N \cdots \tilde A_1,\quad \tilde A_n = \tfrac12 \overline{A^{(n)}} + \tfrac12 I$ | 모든 층을 거친 입력→`<call>` 흐름. residual 경로를 $\tfrac12 I$로 넣음 |

- **가중치가 크다고 영향이 큰 것은 아니다.** 가중치를 많이 받아도 보내는 값 벡터가 작으면 영향이 작다.
  그래서 가중치 옆에 norm을 같이 본다(Kobayashi 외, 2020).
- rollout은 FFN과 LayerNorm을 무시하는 근사다(Abnar & Zuidema, 2020).

## 결과: `draft` 사례

| | 2층에서 `<call>`이 가장 많이 가져오는 토큰 (norm) | `draft`에 주는 가중치 (헤드 평균) |
|---|---|---|
| A (정답) | `log` 120.6 — 요청 단어 | 0.00 |
| B (오답) | `draft` 115.7 — 1위 | 0.75 (헤드 4개 중 3개가 1.00) |

- 정답 문장에서는 2층 attention이 요청 단어 `log`에서 정보를 가져온다.
- 명사 하나를 `draft`로 바꾸면, 같은 2층 attention이 `draft`로 넘어간다.
- 층별 hidden state 비교에서 본 "2층 attention에서 오답으로 넘어감"과 같은 지점이다.
- rollout도 `draft`로 가는 흐름이 0.02에서 0.42로 커진다.

### 노트의 j* 와 ΔA ([`Interpretability math.pdf`](../Interpretability%20math.pdf) 6쪽)

$$
j^* = \arg\max_j A[t, j], \qquad \Delta A[t, :] = A_{F}[t, :] - A_{N}[t, :]
$$

| 층 | j* (A, 정답) | j* (B, 오답) | ΔA가 가장 큰 토큰 |
|---|---|---|---|
| L1 | `log` 0.39 | `log` 0.28 | `draft` +0.22 |
| L2 | `log` 0.58 | **`draft` 0.75** | **`draft` +0.75** |

(가중치는 헤드 평균. 1층에서는 B도 아직 `log`를 가장 많이 보지만, `draft`로 가는 가중치가 가장 크게 늘었다.)

## 결과: 모든 오판 사례 (`--summary`)

바뀐 단어로 가는 attention (B − A), 판정이 뒤집힌 치환과 그대로인 치환 비교:

| | 뒤집힌 치환 (207건) | 그대로인 치환 (69건) |
|---|---|---|
| 1층 가중치 변화 | +0.218 | −0.001 |
| 2층 가중치 변화 | +0.258 | −0.030 |
| 2층 norm (B) | 60.92 | 2.14 |
| 1층에서 바뀐 단어가 1위 (norm) | 52% | 0% |
| 2층에서 바뀐 단어가 1위 (norm) | 57% | 0% |

판정을 뒤집는 치환은 `<call>`의 attention을 바뀐 단어 쪽으로 끌어온다. 판정이 그대로인 치환은 그렇지 않다.

## 실행

```bash
python attention_analysis.py                    # draft 사례
python attention_analysis.py --summary          # 모든 오판 사례
python attention_analysis.py --augment --summary
python attention_analysis.py --pair "문장 A" "문장 B"
```
