# PETR4 — comparação de três modelos (28/11/2026)

**Execução auditável:** [GitHub Actions #46170604821](https://github.com/TheTRIBE-HighTECH/elliot_waves/actions/runs/46170604821), artefato `PETR4-three-models-46170604821` (ID 15272908386, retenção 30 dias).

## Protocolo

Fonte fixada no commit `30e18599e1283ccb77a152fce55d3d5dea1f646a` de `msbrasil/Series-Temporais-PETR4`: PETR4, Ibovespa, Brent em dólares e reais, USD/BRL, volume e preço ajustado. Existe um candle OHLC inconsistente em 02/04/2014, excluído na carga.

**Períodos:** 2014–2023 formação; 2024 validação; 02/01/2025 a 17/11/2026 teste, 469 pregões. Capital R$ 10.000, 13 pontos-base por lado, comprado ou em caixa, sem alavancagem. Entrada/saída sempre na abertura do pregão seguinte após avaliação de fechamento; nenhum stop intradiário fictício. O retorno ajustado por proventos é uma **proxy**, não uma reconciliação executável de dividendos, impostos e custos reais B3. O caixa não rende CDI.

## As três estratégias

**A · Rompimento ATR.** Resistência: máxima dos 10 pregões anteriores; compra se `close_t > resistance_{t-1} + 0.35*ATR7_t`. Após compra, trailing stop: `close_t < peak_high_since_entry - 1.5*ATR7_t`. Todos os sinais são de fechamento e executados no próximo open. Entre 192 alternativas, estes parâmetros foram escolhidos **somente em 2024**. Retorno de validação +21,705%, máximo drawdown 8,875%, 6 entradas.

**B · MACD intermercados.** `MACD_EMA(12,35) > Signal_EMA5` E `(Ibovespa defasado > SMA50 OU BrentBRL defasado > SMA50)`. Sinais externos têm pelo menos um pregão de atraso para evitar confusão de horários internacionais. **Importante:** os parâmetros MACD foram encontrados anteriormente após observar 2025–2026; o resultado nesta amostra é reprodução retrospectiva e não teste independente.

**C · ATR + Elliott GRU multiescala.** Mesmas regras de A, exigindo confirmação de uma GRU três-cabeças (17 classes, reversão e confiança) treinada exclusivamente em 2014–2023, com seleção de pesos em 2024. Confirmação pela soma de `P(bull_1)+P(bull_3)+P(bull_5)` ≥ 0,616113, reversão ≤0,70, confiança ≥0,10. O corte 0,616113 foi o quantil 40% das probabilidades nos exemplos de rompimento de **2024**, sem ajuste para rentabilidade de 2025–2026. Ambos os relógios compartilham um único checkpoint PyTorch; o informacional é baseado em **volume**, não em grafo de notícias Petrobras.

O modelo Elliott aprendeu com **pseudo-rótulos retrospectivos de pivôs**, não contagens humanas verificadas. A qualidade continua baixa: no teste, F1 macro ≈0,1299 (cronológico) e ≈0,1081 (informacional).

## Resultados reais de backtest — série ajustada

| Modelo | 2025 | 2026 até 17/11 | Total | Máximo DD | Sharpe | Entradas | Exposição |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATR + Donchian | −19,83% | +13,44% | **−9,05%** | 20,53% | −0,34 | 15 | 21,75% |
| MACD + Brent/Ibovespa | +4,63% | +65,06% | **+73,94%** | 13,04% | 1,64 | 19 | 49,47% |
| ATR + Elliott neural cronológico | −6,31% | +11,27% | **+4,25%** | 15,47% | 0,19 | 21 | 14,29% |
| ATR + Elliott neural informacional | −7,12% | +9,27% | **+1,48%** | 11,40% | 0,07 | 26 | 14,07% |
| Buy and Hold | −6,00% | +50,69% | **+42,65%** | 27,03% | 0,69 | 1 | 100% |

Vantagem aritmética MACD vs Buy & Hold: **+31,28 pontos percentuais**. Intervalo de 95% em bootstrap pareado com blocos de dez pregões para o excesso de **log-retorno acumulado**: **[−37,73; +86,89] pontos percentuais**, incluindo zero. A descoberta original do MACD analisou milhares de combinações; o bootstrap **não** corrige essa seleção múltipla e não demonstra alpha estatisticamente significativo.

C acrescentou +13,30 pontos percentuais à estratégia A sob o relógio cronológico, mas ficou −38,40 pontos abaixo do Buy & Hold. O relógio informacional não melhorou o resultado.

## Arquivos e reprodução

```bash
python -m pip install 'numpy>=1.26,<3' 'pandas>=2,<3'
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python research/petr4_three_models.py --epochs 8 --fee 13 --out research/three_model_runs
```

O artefato contém `report.json`, `paired_neural_predictions.csv`, `petr4_multiscale_gru.pt`, cinco `equity_*.csv`, séries históricas originais e log do treinamento.

Regras baseadas em bibliografia pública do MQL5 (não no código-fonte fechado do Gold Trade Pro):
- <https://www.mql5.com/en/articles/19459> — rompimentos ajustados por volatilidade e ATR.
- <https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_macd> — sinal MACD.
- <https://www.mql5.com/en/code/262> — exemplo oficial Expert Advisor MACD.

## Pendências antes de execução real

Checar uma fonte primária B3 e calendário de proventos, spreads, lote padrão e fracionário, imposto e comissão; remunerar caixa pelo CDI; validação prospectiva 2027 e em outros ativos, sem reajustar os parâmetros. As probabilidades Elliott precisam de supervisão humana para aferir capacidade de reconhecimento de ondas. Um modelo proprietário Gold Trade Pro verdadeiro **não** foi instalado nem executado.
