# `score_and_grad` — IG 적분 안의 기울기를 계산하는 함수

`score_and_grad`는 `tool_attribution.py`에 있는 함수다.
IG 수식에서 적분 안에 들어가는 **기울기 ∂S(t)/∂t를 실제로 계산하는 코드**다.
수식을 문자 그대로 옮긴 것이 아니라, 그 값을 역전파(backpropagation)로 구한다.

```python
def score_and_grad(model, X_token, k, tool_id, rival_id=None):    # tool_attribution.py
    logits, _, caches = model.forward_embedded(X_token)   # ① t를 넣고 forward
    z = logits[k]                                          #    <call> 위치 logit
    ...                                                    # ② S(t) 계산과 ∂S/∂logits (G)
    GX = G @ model.p["E"]                                  # ③ LM head 역전파
    for n in reversed(range(model.N)):                     # ④ Transformer 블록 역전파
        GX, _ = model.block_backward(GX, caches[n], n)
    return s, GX                                           # ⑤ (S(t), ∂S(t)/∂t)
```

②에서 점수 S와 그 미분 G는 `--score` 옵션에 따라 다르다.

| 점수 | S(t) | G = ∂S/∂logits (`<call>` 행만) |
|---|---|---|
| `logp` | log softmax(z)[tool] | onehot(tool) − softmax(z) |
| `margin` (기본) | z[tool] − z[rival] | onehot(tool) − onehot(rival) |

## 수식과의 대응

$$
IG_j = (X_j - X'_j) \int_0^1
\underbrace{\left. \frac{\partial S(t)}{\partial t_j} \right|_{t = X' + \alpha (X - X')}}_{\texttt{score\_and\_grad(model, t, ...)}}
\, d\alpha
$$

| 수식 | 코드 |
|---|---|
| X' | `Xp` (기준점: 요청 단어 임베딩 0, `<bos>`·`<call>`은 그대로) |
| α | `alpha` |
| t = X' + α(X − X') | `t = Xp + alpha * (X - Xp)` (`attribute` 함수 안) |
| S(t) | `score_and_grad`가 돌려주는 첫 번째 값 `s` |
| ∂S(t)/∂t | 두 번째 값 `GX`, 호출하는 쪽에서는 `g_t` |
| ∫₀¹ … dα | `for alpha in ...: acc += g_t` 후 `acc / steps` (리만 합) |

IG 계산 부분(`attribute` 함수):

```python
for alpha in (np.arange(steps) + 0.5) / steps:
    t = Xp + alpha * (X - Xp)                  # point on the path: t = X' + alpha (X - X')
    _, g_t = score_and_grad(model, t, k, tool_id, rival_id)   # dS(t)/dt
    acc += g_t
ig = ((X - Xp) * (acc / steps)).sum(1)
```

## 인자

`score_and_grad(model, t, ...)`의 `...`는 문서에서 줄여 쓴 것이다. 실제 코드에서는 다음이 들어간다.

| 인자 | 뜻 |
|---|---|
| `model` | 학습된 TinyGPT |
| `X_token` | 입력 임베딩. IG에서는 경로 위의 점 `t`를 넣는다 |
| `k` | `<call>` 토큰의 위치 (점수를 읽는 행) |
| `tool_id` | 설명할 도구 |
| `rival_id` | 비교 대상 도구. `None`이면 `logp`, 값이 있으면 `margin` |

함수 이름은 **score**(점수 S)와 **grad**(그 점수의 기울기)를 함께 돌려준다는 뜻이다.

## 학습할 때의 역전파와 다른 점

| | 학습 (`loss_and_backward`) | attribution (`score_and_grad`) |
|---|---|---|
| 미분하는 값 | loss | 점수 S |
| 시작 기울기 | (P − Y) / T | 위 표의 G (`<call>` 행만) |
| 쓰는 기울기 | 파라미터 기울기 | 입력 임베딩 기울기 ∂S/∂t |
| 블록 역전파 | `block_backward` | 같은 `block_backward` (파라미터 기울기는 버림) |

역전파가 맞는지는 `python tool_attribution.py --check`로 확인한다.
입력 임베딩을 ±ε만큼 흔든 수치미분과 비교하며, margin 기준 상대오차는 약 1.5e-9다.
