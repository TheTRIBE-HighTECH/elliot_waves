# Elliott–Fibonacci × Relógio Informacional — protótipo v0.1

Implementação reprodutível de um **teste de hipótese**, não de um sistema comprovado de previsão financeira.

## Deploy no Vercel

Importe o repositório com o **Root Directory** na raiz. O build estático monta o painel interativo em `/` e a aba de pesquisa em `/backtest/`. O Vercel encaminha as solicitações de arquivos históricos ao domínio público da Binance.

Para atualizar o painel publicado, execute `python modelo.py --out demo_output` e faça commit dos arquivos gerados que deseja publicar.

## Backtest spot

Em **Backtest**, escolha BTCUSDT, ETHUSDT ou SOLUSDT, selecione um intervalo de 1m a 1d e informe as datas UTC inclusivas. O formulário baixa arquivos mensais ou diários da Binance. Também aceita um CSV com `timestamp,open,high,low,close,volume,quote_volume,num_trades,taker_buy_volume`. O período selecionado deve estar completo, sem duplicidades, e tem limite de 100.000 candles. Timestamps representam a abertura da vela. A fonte e os hashes dos arquivos e da amostra canônica ficam no JSON exportado.

O motor Python roda em um Web Worker via Pyodide. O navegador precisa de acesso ao CDN do Pyodide durante a inicialização. A execução pode ser cancelada e o JSON exportado contém ID, parâmetros, versão, proveniência, sinais, ordens, trades, saldos, patrimônio, métricas da validação e logs dos estágios. Não há fila de jobs ou persistência no servidor. Exporte o JSON para arquivar a execução. Use a CLI para executar fora do navegador:

```bash
python -m pip install -r requirements.txt
python -m backtest.experiment --symbol BTCUSDT --interval 1h \
  --start 2025-01-01 --end 2025-06-30 --out btc-backtest.json
python -m unittest -v
node backtest/build.mjs
```

A CLI aceita `--csv` e `--options` (arquivo JSON com ajustes descritos em `backtest/experiment.py`). O build precisa de Node.js e gera `.site/`.

### Execução e comparação

- Os modelos A/B usam cruzamentos de momento em janelas de duração equivalentes nos respectivos relógios como controle sem geometria Elliott. C/D usam a entrada na possível onda 3, com limiar configurável do pivô e stop abaixo do fundo da onda 2. A onda 2 precisa manter o fundo acima da origem da onda 1. O alvo padrão é o fundo da onda 2 mais 1,618 vezes a diferença entre o topo e a mínima do pivô inicial da onda 1. Depois de confirmar a onda 2, o sinal exige o primeiro cruzamento em fechamento acima da maior máxima da onda 1. Uma mínima abaixo do fundo da onda 2 invalida a configuração, inclusive no candle do rompimento. A execução deve permanecer acima do nível rompido e abaixo do alvo. Quando ativada, a duração mínima da onda 1–2 é medida no relógio de cada modelo. D pode usar o filtro adicional de velocidade. O filtro de Hurst é opcional em todos os quatro modelos.
- O relógio reutiliza a transformação exponencial e os z-scores retrospectivos do projeto. Intensidade de trades, volume e desequilíbrio são proxies de atividade. As bandas Fibonacci do Hurst são avaliadas em unidades de um candle nominal, em janelas causais de 256 barras amostradas a cada 16 barras.
- O sinal só surge no fechamento que confirma o pivô. A ordem a mercado utiliza a abertura seguinte, spread e slippage de cada lado, mais taxa taker. A alternativa maker fica pelo próximo candle no fechamento anterior, exige abertura acima da ordem e mínima abaixo dela, e expira sem execução se não houver cruzamento. Se a entrada maker e o alvo ocorrerem no mesmo candle, o alvo só é elegível no candle posterior. O stop pressupõe execução nessa vela se atravessado. Para ordens a mercado, a política padrão quando stop e alvo são atingidos na mesma vela é stop primeiro. Em gaps de saída a execução parte da abertura. Custos e risco até o stop entram no dimensionamento inicial, mas gaps podem exceder o risco de 1%. O bruto preserva as quantidades e os preços de referência da simulação líquida. O mercado spot opera comprado e mantém uma posição por vez.
- Os modelos usam as mesmas janelas, tamanho de grade, condições de execução e custos. A seleção é feita por retorno líquido menos drawdown máximo no treino. Validação e teste usam parâmetros congelados. As janelas walk-forward ficam no período de desenvolvimento, antes do teste final. Cada janela refaz a seleção com dados anteriores e o embargo configurado. Nenhuma posição ou ordem cruza os limites de um segmento.
- Os testes reportam dispersão relativa das durações do mesmo conjunto de impulsos (H1) e diferença de retornos diários líquidos de D frente aos controles (componente econômico de H3). A precisão direcional a cinco barras é descritiva, usa sinais completos e evita janelas sobrepostas. Os intervalos de confiança usam bootstrap em blocos e podem ficar indisponíveis para amostras curtas. As razões de preço dos mesmos pivôs são invariantes à mudança de eixo. H2 permanece inconclusiva até que um detector multiescala independente faça a identificação de padrões em cada relógio.
- O score geométrico exibe cinco componentes brutos e percentis calibrados exclusivamente no treino. É uma medida descritiva, sem interpretação probabilística. A variante ABC implementa somente a correção simples após um impulso confirmado. A soma fractal da proposta, subondas recursivas e futuros não fazem parte desta versão. Evite interpretar os Sharpe e Sortino anualizados como testes de significância, especialmente para janelas curtas.

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
