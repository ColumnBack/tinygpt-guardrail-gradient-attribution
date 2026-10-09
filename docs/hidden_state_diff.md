# 층별 hidden state 비교 (layer-wise hidden-state diff)

IG가 **어떤** 입력 토큰이 결정을 만들었는지 알려 준다면, 이 방법은 정답 입력과 오답 입력이
**몇 번째 층에서** 갈라지는지 알려 준다. 코드는 `hidden_state_diff.py`이고, `tinygpt.py`는 읽기만 한다.

## 비교하는 두 입력

`long_sentence_study.py`의 오판 사례 중, 상황 설명의 명사 **하나만** 다른 두 문장을 쓴다.
길이와 위치가 같아서 위치별로 바로 비교할 수 있다.

```
A: the server shows an error so help me read the log file   -> filesystem.read_file  (정답)
B: the draft  shows an error so help me read the log file   -> filesystem.write_file (오답)
```

## 단계

`tinygpt.block_forward`가 남기는 값으로 각 단계의 hidden state를 꺼낸다.

$$
\begin{aligned}
h^{(0)} &= E[\text{ids}] + P && \text{(embed)} \\
Y^{(n)} &= \mathrm{LN}_1\big(X^{(n)} + \mathrm{Attn}(X^{(n)})\big) && \text{(L}n\text{.attn)} \\
X^{(n+1)} &= \mathrm{LN}_2\big(Y^{(n)} + \mathrm{FFN}(Y^{(n)})\big) && \text{(L}n\text{.ffn)}
\end{aligned}
$$

마지막 단계가 출력층이 읽는 H이고, $Z = H E^\top$ 이다.

## 세 가지 측정

**1. 위치별 상대 차이**

$$
d_s(i) = \frac{\lVert h_B^{(s)}[i] - h_A^{(s)}[i] \rVert}{\lVert h_A^{(s)}[i] \rVert}
$$

embed 단계에서는 바뀐 단어 위치만 차이가 있다. 층을 지나며 attention이 그 차이를 다른 위치로 옮긴다.
결정을 내리는 `<call>` 위치에 차이가 언제 도착하는지 본다.

**2. logit lens**

각 단계의 `<call>` hidden state를 출력층에 바로 통과시킨다.

$$
z^{(s)} = h^{(s)}[k]\, E^\top, \qquad m^{(s)} = z^{(s)}[\text{정답}] - z^{(s)}[\text{오답}]
$$

$m^{(s)} > 0$ 이면 그 단계에서 정답 도구가 앞선다. B에서 **그 단계부터 끝까지 계속 $m < 0$인 첫 단계**를
"오답으로 넘어간 단계"로 센다. 중간층은 출력층으로 읽히도록 학습된 것이 아니어서 근사적인 보기이고,
마지막 단계만 실제 출력이다.

**3. 어느 부품이 차이를 옮겼나**

블록마다 `<call>` 위치에서 attention 출력과 FFN 출력이 A와 B 사이에 얼마나 다른지 비교한다.

$$
\lVert \mathrm{Attn}_B[k] - \mathrm{Attn}_A[k] \rVert \quad \text{vs} \quad \lVert \mathrm{FFN}_B[k] - \mathrm{FFN}_A[k] \rVert
$$

## 결과: `draft` 사례

| 단계 | `<call>` 상대 차이 | logit lens A | logit lens B |
|---|---|---|---|
| embed | 0.00 | −0.05 | −0.05 |
| L1.attn | 0.32 | 1.90 | 0.81 |
| L1.ffn | 0.27 | 2.02 | 1.19 |
| **L2.attn** | **0.95** | 7.88 | **−1.73** ← 여기서부터 오답 |
| L2.ffn | 1.47 | 21.04 | −22.24 |

- 1층을 지나도 B는 아직 정답 도구 쪽이다.
- **2층 attention**에서 `draft`의 정보가 `<call>`로 옮겨지면서 오답으로 넘어가고, 2층 FFN이 그 차이를 크게 키운다.
- 부품별 변화: 1층은 FFN(39.98)이 attention(6.96)보다 크고, 2층은 attention(133.38)이 가장 크다.
- 다음 단계인 attention 분석에서 "2층에서 `<call>`이 `draft`를 얼마나 보는지"를 확인할 근거가 된다.

## 결과: 모든 오판 사례 (`--summary`)

오답으로 넘어간 단계의 분포:

| 단계 | 처음 모델 (207건) | 보강 후 모델 (69건) |
|---|---|---|
| L1.attn | 74 | 38 |
| L1.ffn | 32 | 4 |
| L2.attn | 77 | 19 |
| L2.ffn | 24 | 8 |

`<call>` 위치의 평균 상대 차이 (처음 모델):

| 단계 | 판정이 뒤집힌 치환 | 판정이 그대로인 치환 |
|---|---|---|
| L1.attn | 0.611 | 0.082 |
| L1.ffn | 0.579 | 0.063 |
| L2.attn | 0.905 | 0.052 |
| L2.ffn | 1.149 | 0.068 |

- 판정을 뒤집는 치환은 `<call>` 위치의 hidden state를 층에 따라 **7~17배** 크게 바꾼다. 판정이 그대로인 치환은 `<call>`까지 거의 닿지 않는다.
- 오답으로 넘어가는 지점은 1층 attention과 2층 attention에 몰려 있다. 결정 위치로 정보를 옮기는 것이 attention이라는 점과 맞는다.

## 실행

```bash
python hidden_state_diff.py                    # draft 사례
python hidden_state_diff.py --summary          # 모든 오판 사례
python hidden_state_diff.py --augment          # 보강 후 모델로
python hidden_state_diff.py --pair "문장 A" "문장 B"   # 같은 길이의 두 문장
```
