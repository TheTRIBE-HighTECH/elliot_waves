# Integração do grafo temporal real ao detector neural Elliott

## Origem verificável

`research/hn_graph.py` pesquisa publicações históricas no Hacker News Search
(Algolia: https://hn.algolia.com/api) e recupera árvores completas via
`/api/v1/items/<story_id>`. As arestas são relações **efetivas de resposta**
(`parent_id` explícito ou extraído da árvore de replies), não inferências a
partir de preço, volume ou correlação temporal.

Um grafo não representa o universo de notícias financeiras: HN é uma amostra
com viés tecnológico, e relações de resposta não comprovam difusão causal.
A coleta histórica é reconstruída a partir do estado da API atual; timestamps
informam a data original da mensagem, mas não provam que a resposta estava
disponível de forma imutável naquela data. Não guardar textos ou nomes.

O coletor guarda `hn_nodes.csv` e `provenance.json`, incluindo URLs originais,
horário da recuperação e SHA-256 por estrutura de thread. A captura de 2020
é um **estudo de caso** de COVID-19, não um treinamento com preços de 2020.

## Dez estatísticas topológicas horárias causais

1. log eventos totais; 2. log comentários; 3. log raízes;
4. log respostas de profundidade >=2; 5. profundidade máxima;
6. profundidade média; 7. concentração de respostas no mesmo pai por hora;
8. atraso médio log entre filho e pai; 9. log de conversas distintas;
10. Herfindahl de respostas por pai.

Os eventos são atribuídos ao candle horário **somente no fechamento da hora**.
Todas as integrações e janelas de amostragem restringem índices a `<=i`.
A velocidade do relógio do grafo usa padronização com observações
**anteriores**, após `shift(1)`, e é estritamente positiva.

## Modelo

`research/graph_fusion.py` estende a TCN multiescala existente (1,3,8)
acrescentando 10 canais de topologia horária às 4 features de preço/volume
de cada escala. É **condicionamento neural por atributos topológicos de grafo**,
não uma Temporal GNN completa com message passing treinável por indivíduo.
Mantém três saídas: 17 classes de ondas, reversão em horizonte de 8 horas
e confiança substituta (sujeita a calibração).

O treinamento usa dados reais BTCUSDT 1h, de 2024–2025. Os rótulos de
ondas são **pseudorrótulos de pivôs retrospectivos**, não contagens humanas
validadas. A comparação testa a MESMA TCN e os MESMOS pesos:

| Modo | Eixo de tempo | Canais de topologia |
| --- | --- | --- |
| calendar_no_graph | Calendário | Zerados |
| calendar_graph | Calendário | Reais |
| graphclock_no_graph | Grafo | Zerados |
| graphclock_graph | Grafo | Reais |

Conjunto temporal: 2024 treino; 2025-H1 validação; 2025-H2 teste.
O relatório separa F1 das pseudo-ondas, Brier de reversão e lucro do backtest;
a seleção da configuração deve ocorrer exclusivamente na validação.
Este protocolo testa um relógio diferente do relógio por volume já existente.
Não produz evidência de causalidade nem confirma Fibonacci nos preços.

## Executar

```bash
pip install -r research/requirements.txt
python -m unittest discover -s research -p 'test_graph.py' -v
python research/download_binance.py --symbols BTCUSDT --start 2024-01 --end 2025-12 --out research/datasets
python research/hn_graph.py --start 2024-01 --end 2025-12 --per-topic-per-month 8 --out research/graph_data
python research/graph_fusion.py --csv research/datasets/BTCUSDT_1h.csv --nodes research/graph_data/hn_nodes.csv --out research/graph_runs
```

O workflow `.github/workflows/temporal-graph.yml` executa tudo em CI e publica
as métricas e pesos como artefatos separados, sem alterar a aplicação Vercel.
Se a API falhar ou nenhum grafo cruzar o período dos candles, interrompe o
experimento em vez de inventar a rede social.

## Próxima evolução necessária

Capturar anotações humanas validadas com timestamps de confirmação; substituir
os pseudorrótulos e acrescentar `label_known_at` às amostras causais.
Adicionar outras fontes com cadeias de transmissão genuínas (e licenças claras),
um grafo placebo preservando atividade por hora mas embaralhando arestas,
múltiplas sementes, IC por blocos e comparação com séries apenas por volume.
Um verdadeiro Temporal Graph Network end-to-end requer embeddings individuais,
message passing condicionado ao evento e backpropagation pelos vínculos.
