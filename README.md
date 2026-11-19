# Elliott–Fibonacci × Relógio Informacional — protótipo v0.1

Implementação reprodutível de um **teste de hipótese**, não de um sistema comprovado de previsão financeira.

## Deploy no Vercel

Importe o repositório com o **Root Directory** na raiz. O `vercel.json` publica os arquivos de `demo_output/` como site estático e abre o painel interativo de demonstração em `/`.

Para atualizar o painel publicado, execute `python modelo.py --out demo_output` e faça commit dos arquivos gerados que deseja publicar.

## Instalação e execução

```bash
python -m pip install -r requirements.txt
python modelo.py --out resultados
```

Sem argumentos de fonte, o programa gera **uma série sintética** e exporta quatro imagens, um HTML interativo autocontido, um CSV com o relógio, um CSV com as estimativas de Hurst e diagnósticos JSON.

Com dados **reais** da Binance, por exemplo:

```bash
python modelo.py --binance-month 2026-10 --symbol BTCUSDT --interval 1m --out resultados_btc
```

Ou para um único dia (a estimativa de Hurst terá menos janelas):

```bash
python modelo.py --binance-day 2026-11-17 --symbol BTCUSDT --interval 1m --out resultados_btc_dia
```

É necessário acesso à Internet para baixar arquivos da Binance. O programa utiliza o acervo oficial https://data.binance.vision/ (dados spot). Dados spot a partir de 2025 usam timestamps em **microssegundos**; o importador reconhece segundos, milissegundos e microssegundos. A origem dos dados de mercado não é responsabilidade deste protótipo.

Use seu próprio CSV:

```bash
python modelo.py --csv meus_dados.csv --out resultados_csv
```

Cabeçalho mínimo (UTC):

```csv
timestamp,close,volume,quote_volume,num_trades,taker_buy_volume
2026-10-01T00:00:00Z,66000,12.3,811800,170,6.8
```

São necessárias 750 ou mais observações regulares; recomenda-se 3.000 ou mais para visualizar mudanças locais. `--max-bars` limita a janela final sem alterar a fonte original.

## Equações implementadas

O relógio operacional é determinado por dados de atividade **observáveis até o fechamento de cada barra**:

- Z-scores robustos de `log(1 + número de trades)`, `log(1 + volume cotado)` e `abs(2 * volume de compras agressoras / volume total - 1)`;
- cada estatística de centralização utiliza apenas o **passado** (mediana e MAD móveis deslocadas em 1 observação);
- `v_t = clip(exp(0.5 * [0.45*z_trades + 0.35*z_volume + 0.20*z_abs_imbalance]), 0.35, 3.5)`;
- `tau_t = soma(v_i * delta_t_i)`.

O processo de preço **não é alterado** na figura 2: são os mesmos fechamentos reorganizados por abscissa informacional. A figura 3 compara estimativas locais de expoentes H entre relógios, em janelas retrospectivas de 700 barras, usando escalas de defasagem Fibonacci:

- curta: 1, 2, 3, 5, 8, 13;
- média: 3, 5, 8, 13, 21, 34;
- longa: 8, 13, 21, 34, 55, 89.

`Var[X(u + Δ) - X(u)] ≈ C Δ^(2H)`. O ajuste linear log–log da variância em função da defasagem estima H. Para o relógio não uniforme, usa interpolação **apenas entre observações passadas**; isso ainda pode introduzir viés de microestrutura nas escalas pequenas.

## Elliott e Fibonacci

O detector simples de pivôs opera com reversão de aproximadamente 0,25% em log-preço. O limite é ajustável com `--reversal 0.0035`. Um pico/vale somente é **conhecido quando confirmado por uma reversão posterior**. A rotina avalia candidatas impulsivas (1–5), respeita restrições convencionais básicas e penaliza desvios de proporções Fibonacci. Isso não prova que Elliott seja válido nem gera sinais de compra/venda. A geometria corretiva ABC e a decomposição recursiva completa ainda não foram implementadas.

## Limitações científicas centrais

1. A intensidade de trades é um **proxy de atividade**, não mede diretamente surpresa de Shannon, transmissão de notícias ou fluxo causal de informação. Notícias em timestamp seriam uma próxima fonte de dados.
2. Os parâmetros do relógio são escolhas de partida e **não foram estimados** ou validados empiricamente.
3. O R² dos ajustes Hurst é descritivo. Não equivale a eficácia preditiva.
4. Não inferir tendência futura de um candidato Elliott sem avaliação fora da amostra e considerando o instante de confirmação.
5. A interpolação em relógio irregular pode afetar o H estimado. Repetir os testes com microdados e estimadores alternativos.
6. Para testar valor preditivo, comparar quatro configurações (fractal livre/Elliott–Fibonacci × relógio cronológico/informacional) em janelas walk-forward sem vazamento.
7. Os gráficos de simulação não são evidência empírica da hipótese.

## Saídas

- `PAINEL_INTERATIVO.html` — painel navegável, com zoom, informação ao passar o mouse e visualização das candidatas Elliott, sem depender de servidor ou internet;
- `PAINEL_COMPLETO.png` — os 3 resultados em um único painel, mais velocidade do relógio;
- `01_preco_cronologico.png`, `02_preco_informacional.png`, `03_espectro_hurst.png` — gráficos individuais;
- `relogio_informacional.csv` — preço, fluxos, desequilíbrio, velocidade, tau;
- `hurst_por_escala.csv` — estimativas locais e R² para ambos os relógios;
- `diagnosticos.json` — resumo de execução e candidatos Elliott confirmados.
