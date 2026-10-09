# IG에서 사용한 식

코드에서 쓴 식은 **1번**이다.

$$
\boxed{ IG_j = (X_j - X'_j) \int_0^1 \left. \frac{\partial S(t)}{\partial t_j} \right|_{t = X' + \alpha (X - X')} \, d\alpha }
$$

`tool_attribution.py`의 `attribute` 함수가 이 식을 그대로 계산한다.

```python
Xp = X.copy(); Xp[1:k] = 0.0                     # X' (기준점)
for alpha in (np.arange(steps) + 0.5) / steps:   # α를 구간 가운데 점으로 샘플링
    t = Xp + alpha * (X - Xp)                    # 경로 위의 점 t = X' + α(X − X')
    _, g_t = score_and_grad(model, t, ...)       # ∂S(t)/∂t
    acc += g_t
ig = ((X - Xp) * (acc / steps)).sum(1)           # (X − X') × 평균 기울기
```

- **적분 근사**: 적분은 구간 가운데 점을 쓰는 리만 합으로 계산한다. `steps`가 클수록 정확하고, 기본값은 64, 진단 예제에서는 256을 썼다.

  $$
  IG_j \approx (X_j - X'_j) \cdot \frac{1}{m} \sum_{k=1}^{m} \left. \frac{\partial S(t)}{\partial t_j} \right|_{t = X' + \alpha_k (X - X')},
  \qquad \alpha_k = \frac{k - 1/2}{m}
  $$

- **마지막 `.sum(1)`**: 식의 j는 원래 입력의 한 차원이다. 여기서는 단어 하나가 64차원 임베딩이라, 단어 i의 64개 차원 IG를 더해 **단어별 점수**로 만든다. completeness는 차원별로도, 단어별로 더해도 성립한다.

  $$
  IG_{\text{token } i} = \sum_{d=1}^{64} IG_{i,d}
  $$

- **기준점 X'**: 요청 단어의 임베딩은 0이고, `<bos>`와 `<call>`은 실제 값 그대로 둔다. 그래서 `<bos>`와 `<call>`은 X − X' = 0이 되어 IG도 0이다.

## 2번 식과의 관계

$$
IG_j(X) = \int_{X'_j}^{X_j} \frac{\partial S(t)}{\partial t_j} \, dt_j
$$

2번은 **경로 적분(path integral)의 일반형**이다. 이 식만으로는 경로가 정해지지 않는다. t_j가 X'_j에서 X_j로 가는 동안 **나머지 차원 t_k가 어디에 있는지** 정해야 값이 정해진다.

- 경로를 **직선**으로 잡으면

  $$
  t(\alpha) = X' + \alpha (X - X'), \qquad dt_j = (X_j - X'_j) \, d\alpha
  $$

  이고, 이를 2번에 대입하면 정확히 **1번이 된다**.
- 즉 2번은 "Path Methods" 전체를 나타내고, IG는 그중 직선 경로를 고른 특별한 경우다. Sundararajan 외(2017) 논문에서도 이렇게 설명한다.
- 직선 경로를 고르는 이유는 몇 가지 바람직한 성질(대칭성 등)을 함께 만족하는 경로이기 때문이다.

2번만 써 두면 경로가 빠져 있어서 식이 덜 정의된 상태다. 그래서 IG를 쓸 때는 1번이나, 경로를 명시한 2번 형태로 쓰는 것이 정확하다.

## completeness

$$
\sum_j IG_j = S(X) - S(X')
$$

모든 차원(또는 모든 단어)의 IG를 더하면 점수 변화 전체와 같다. 코드 출력의 `completeness` 줄이 이 값을 확인한다.
