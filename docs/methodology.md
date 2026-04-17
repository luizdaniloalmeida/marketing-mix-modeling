# Metodologia Técnica do Marketing Mix Modeling

## O que é Marketing Mix Modeling?

Marketing Mix Modeling (MMM) é uma técnica estatística que mede o impacto real de cada canal de marketing na receita de uma empresa. A ideia é simples: separar o que é vendido "naturalmente" (sem nenhuma mídia ativa) do que foi gerado por cada canal como Meta Ads, Google Ads, Email e assim por diante.

Na prática, isso significa pegar o histórico de investimento semanal em cada canal e usar regressão para descobrir quanto cada real investido contribuiu para a receita. Diferente de atribuição baseada em cookies (que já morreu com as restrições de privacidade), o MMM trabalha com dados agregados e não depende de rastreamento individual.

Dois frameworks de referência popularizaram essa abordagem:

- **Meridian** (Google, 2024): framework open-source em Python/JAX que usa abordagem bayesiana para MMM, com priors informativos e intervalos de credibilidade.
- **Robyn** (Meta, 2022): framework em R que automatiza a calibração de hiperparâmetros com algoritmos evolutivos (Nevergrad).

Este projeto segue a mesma base conceitual desses frameworks, mas com implementação própria em Python, focando em clareza e didática.

## Transformações de Mídia

Antes de jogar os dados no modelo, precisamos tratar dois fenômenos que acontecem em qualquer campanha de marketing:

### 1. Adstock (Carryover)

Quando você vê um anúncio na segunda-feira, a marca não desaparece da sua cabeça na terça. Parte do efeito permanece por dias ou semanas. Isso é o carryover, e o adstock geométrico captura esse comportamento.

**Fórmula:**

```
adstock(t) = spend(t) + decay * adstock(t-1)
```

Onde `decay` é um valor entre 0 e 1 que controla quanto do efeito persiste. Na prática:

- **Canais de branding** (conteúdo orgânico, LinkedIn) têm decay alto (0.6 a 0.7). O efeito demora pra sumir.
- **Canais de performance** (Google Ads, email) têm decay baixo (0.2 a 0.3). O impacto é quase imediato.

### 2. Saturação de Hill

Dobrar o investimento em um canal não dobra a receita. A partir de certo ponto, cada real adicional rende menos. Isso é o efeito de retornos decrescentes, e a curva de Hill modela esse comportamento.

**Fórmula:**

```
saturação(x) = 1 / (1 + (K / x)^S)
```

Onde:
- `K` (half-saturation): nível de investimento onde a resposta atinge 50% do máximo.
- `S` (slope): controla quão rápido o canal satura.

O resultado é um valor entre 0 e 1 que representa o "grau de saturação" do canal. Perto de 0, o canal tem muito espaço pra crescer. Perto de 1, já saturou.

### Pipeline de transformação

Para cada canal, o processo é:

```
spend bruto → adstock geométrico → saturação de Hill → feature para o modelo
```

## O Modelo

### Regressão OLS

O modelo é uma regressão linear simples (OLS) onde:

```
receita = intercepto + Σ(coef_canal × canal_transformado) + Σ(coef_contexto × variável_contexto) + erro
```

As variáveis de contexto são: índice de sazonalidade, feriados, Black Friday, pressão competitiva e tendência temporal. Elas controlam fatores externos que afetam a receita independentemente do marketing.

### Restrição de não-negatividade

Um coeficiente negativo para um canal de marketing significaria que investir nele *reduz* a receita. Isso não faz sentido. Quando o OLS retorna algum coeficiente negativo nos canais, ativamos um fallback: reajustamos o modelo usando `scipy.optimize.minimize` (método L-BFGS-B) com bounds `[0, +∞)` para os canais, enquanto intercepto e variáveis de contexto permanecem livres.

### Diagnósticos

Para validar o modelo, verificamos:

| Métrica | O que mede | Valor desejado |
|---------|-----------|----------------|
| R² | Proporção da variância explicada | ≥ 0.80 |
| MAPE | Erro percentual médio | < 10% |
| Durbin-Watson | Autocorrelação dos resíduos | ~2.0 |
| VIF | Multicolinearidade entre features | < 10 por feature |

Além disso, inspecionamos graficamente: QQ-plot dos resíduos (normalidade), resíduos vs. preditos (homocedasticidade) e valores reais vs. preditos (aderência geral).

### Decomposição e ROI

Com os coeficientes estimados, decompomos a receita:

- **Contribuição do canal** = coeficiente × valor transformado (somado ao longo das semanas).
- **Baseline** = intercepto + variáveis de contexto. Representa as vendas que aconteceriam sem nenhum marketing ativo.
- **ROI** = contribuição total / investimento total. Quanto a empresa recebeu de volta para cada real investido.

## Otimização de Budget

O otimizador recebe o modelo treinado e encontra a distribuição de budget que maximiza a receita prevista. Usa SLSQP (Sequential Least Squares Programming) com:

- Restrição de igualdade: soma dos canais = budget total.
- Bounds por canal: pisos mínimos (contratos, presença obrigatória) e tetos máximos (capacidade operacional).

Para prever a receita de uma alocação, assumimos regime estacionário: se o investimento semanal for constante, o adstock converge para `spend / (1 - decay)`.

## Limitações

Este projeto foi feito com dados sintéticos para fins de portfólio. Algumas limitações importantes:

1. **Dados simulados**: os dados foram gerados com parâmetros conhecidos. Em dados reais, a calibração dos hiperparâmetros (decay, half-saturation, slope) precisa ser feita via validação cruzada ou abordagem bayesiana.

2. **Sem interações cross-channel**: o modelo trata cada canal de forma independente. Na realidade, ver um anúncio no Instagram pode influenciar uma busca no Google.

3. **Sem efeitos de longo prazo**: adstock geométrico captura carryover de curto prazo (semanas), mas não modela construção de marca ao longo de meses ou anos.

4. **Causalidade vs. correlação**: regressão mede associação, não causalidade. Para inferência causal mais robusta, seria necessário usar experimentos (geo-lift tests) ou métodos como variáveis instrumentais.

5. **Estacionaridade**: o otimizador assume que os coeficientes do modelo permanecem estáveis ao longo do tempo. Na prática, o modelo deve ser recalibrado periodicamente.

## Referências

### Papers

- Jin, Y., et al. (2017). *Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects*. Google Inc.
- Chan, D. & Perry, M. (2017). *Challenges and Opportunities in Media Mix Modeling*. Google Inc.

### Frameworks

- [Meridian](https://github.com/google/meridian) (Google, 2024) — MMM bayesiano em Python/JAX.
- [Robyn](https://github.com/facebookresearch/Robyn) (Meta, 2022) — MMM automatizado em R.
- [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) (PyMC Labs) — MMM bayesiano com PyMC.

### Conceitos

- Hill, A.V. (1910). *The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves*. Journal of Physiology.
- Broadbent, S. (1979). *One at a Time: A Media Scheduling Model*. Sobre adstock geométrico.
