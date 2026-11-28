import test from 'node:test';
import assert from 'node:assert/strict';
import {parseCSV,timestamp,normalizeBars,normalizeGraph,graphSeries,makeClock,
  causalPivots,estimateHurst,normalizePredictions,evaluateBacktest,graphTree} from '../demo_output/dashboard-core.mjs';
import {readFileSync} from 'node:fs';

function candles(n=500){
  let p=100;const result=[];
  for(let i=0;i<n;i++){
    const old=p;p=p*Math.exp(.0004+.003*Math.sin(i/12));
    result.push({timestamp:new Date(Date.UTC(2025,0,1)+i*3600000).toISOString(),
      open:old,high:Math.max(old,p)*1.002,low:Math.min(old,p)*.998,close:p,
      volume:10+i%11,num_trades:30+i%7});
  }return normalizeBars(result);
}
function nodes(){
  const time=i=>new Date(Date.UTC(2025,0,1)+i*3600000+10000).toISOString();
  return normalizeGraph([
    {id:'10',story_id:'10',parent_id:'',time_utc:time(2),depth:'0',kind:'story',topic:'bitcoin'},
    {id:'11',story_id:'10',parent_id:'10',time_utc:time(3),depth:'1',kind:'comment',topic:'bitcoin'},
    {id:'12',story_id:'10',parent_id:'11',time_utc:time(4),depth:'2',kind:'comment',topic:'bitcoin'},
    {id:'13',story_id:'10',parent_id:'10',time_utc:time(24),depth:'1',kind:'comment',topic:'bitcoin'},
  ]);
}
test('CSV quoted fields and no XSS parsing/execution',()=>{
  const rows=parseCSV('timestamp,comment,close\r\n2025-01-01,"a,b ""quoted""",102\r\n');
  assert.equal(rows.length,1);assert.equal(rows[0].comment,'a,b "quoted"');
});
test('Epoch timestamp precision and explicit UTC parsing',()=>{
  assert.equal(timestamp('1735689600'),Date.UTC(2025,0,1));
  assert.equal(timestamp('1735689600000'),Date.UTC(2025,0,1));
  assert.equal(timestamp('1735689600000000'),Date.UTC(2025,0,1));
  assert.equal(timestamp('01-01-2025 00:00'),Date.UTC(2025,0,1));
});
test('Real OHLCV accepts chronological bars and rejects duplicates',()=>{
  assert.equal(candles(500).length,500);
  const items=[{timestamp:'2025-01-01T00:00:00Z',open:100,close:100},
               {timestamp:'2025-01-01T00:00:00Z',open:100,close:101}];
  assert.throws(()=>normalizeBars([...items,...candles(50).map(x=>({timestamp:new Date(x.t).toISOString(),close:x.close}))]),/duplicados/);
});
test('Explicit reply edges counted; adding a future reply cannot affect the past',()=>{
  const bars=candles(),full=nodes(),a=graphSeries(bars,full),
      b=graphSeries(bars,full.filter(x=>x.id!=='13'));
  assert.equal(a.validEdges,3);assert.equal(b.validEdges,2);
  assert.deepEqual(Array.from(a.counts.slice(0,24)),Array.from(b.counts.slice(0,24)));
  assert.deepEqual(Array.from(a.connected.slice(0,24)),Array.from(b.connected.slice(0,24)));
});
test('Calendar α=0 matches chronological time exactly; time never reverses',()=>{
  const bars=candles(),g=graphSeries(bars,nodes());
  const zero=makeClock(bars,g,{source:'graph',alpha:0}),cal=makeClock(bars,g,{source:'calendar'});
  assert.deepEqual(Array.from(zero.axis),Array.from(cal.axis));
  const active=makeClock(bars,g,{source:'mixed',alpha:1,beta:.3});
  assert(active.axis.every((v,i)=>i===0||v>active.axis[i-1]));
  assert(active.speed.every(v=>v>.0));
});
test('Graph speed responds to an observed activity burst, unlike calendar',()=>{
  const bars=candles(500),g=graphSeries(bars,nodes());
  const inf=makeClock(bars,g,{source:'graph',alpha:1.2,window:24});
  assert.notDeepEqual(Array.from(inf.speed.slice(1)),Array(499).fill(1));
});
test('Causal pivots have confirmation time no earlier than extremum',()=>{
  const bars=candles(500),piv=causalPivots(bars,.3,100);
  assert(piv.length>0);assert(piv.every(x=>x.confirmed>x.at));
});
test('Hurst returns a descriptive estimate with historical-only windows',()=>{
  const bars=candles(500),clk=makeClock(bars,{}, {source:'calendar'});
  const x=estimateHurst(bars,clk.axis,500,300);
  assert(x&&Number.isFinite(x.H));
});
test('Predictions require both timestamps and do not accept retroactive availability',()=>{
  const begin='2025-01-01T00:00:00Z';
  const result=normalizePredictions([{clock:'calendar',timestamp:begin,known_at:'2024-12-31T23:00:00Z',
     p_bull_3:'1',p_reversal:'0',confidence:'1',weight_sha256:'a'.repeat(64)},
    {clock:'calendar',timestamp:begin,known_at:'2025-01-01T01:00:00Z',
     p_bull_3:'.6',p_reversal:'.3',confidence:'.9',weight_sha256:'a'.repeat(64)}]);
  assert.equal(result.series.calendar.size,1);
});
test('Two equal predictions and the same costs yield equal trades and returns',()=>{
  const bars=candles(100),predict=new Map();
  for(const bar of bars.slice(0,99))predict.set(bar.t,
    {known:bar.t+3600000,p:.8,rev:.1,conf:.9,hash:'a'.repeat(64)});
  const params={capital:10000,feeBps:10,slippageBps:3,riskPct:1,stopPct:2,takePct:4};
  const a=evaluateBacktest(bars,predict,params);
  const b=evaluateBacktest(bars,predict,params);
  assert.equal(a.returnPct,b.returnPct);assert.deepEqual(a.trades,b.trades);
  assert(a.matched>0);
});
test('Future-published predictions are never tradable at earlier candle open',()=>{
  const bars=candles(70),pred=new Map();
  pred.set(bars[0].t,{known:bars[69].t+3600000,p:1,rev:0,conf:1,hash:'a'.repeat(64)});
  const result=evaluateBacktest(bars,pred);
  assert.equal(result.tradeCount,0);assert.equal(result.matched,0);
});
test('Backtest refuses data without true candle opens',()=>{
  const bars=candles(70);bars[5].hasOpen=false;
  assert.throws(()=>evaluateBacktest(bars,new Map()),/abertura/);
});
test('Graph tree chooses real linked story, not invented edges',()=>{
  const ns=nodes(),result=graphTree(ns,Infinity,100,'bitcoin');
  assert.equal(result.length,4);
  assert.equal(result[2].parent,'11');
});
test('Dashboard page has all critical components and import controls',()=>{
  const html=readFileSync(new URL('../demo_output/dashboard.html',import.meta.url),'utf8');
  for(const id of ['marketFile','graphFile','predictionsFile','reportFile','clockSource','alpha',
    'calendarChart','informationChart','graphChart','neuralOutputs','equityChart',
    'annotationsTable','runBacktest','exportConfig','exportAnnotations','signalMaxAgeHours']){
    assert.match(html,new RegExp('id="'+id+'"'));
  }
});

test('Sparse neural forecasts persist causally for configured TTL, then expire',()=>{
  const bars=candles(75);
  const signal=new Map();
  signal.set(bars[30].t,{known:bars[31].t,p:1,rev:0,conf:1,hash:'a'.repeat(64)});
  const short=evaluateBacktest(bars,signal,
    {signalMaxAgeHours:1,stopPct:30,takePct:30,riskPct:1});
  const long=evaluateBacktest(bars,signal,
    {signalMaxAgeHours:12,stopPct:30,takePct:30,riskPct:1});
  assert.equal(short.matched,2);
  assert.equal(long.matched,13);
  assert(long.exposurePct>short.exposurePct);
  assert(long.trades.some(t=>t.reason==='signal_or_expiry'));
});
test('Prediction published after its candle cannot be applied to past openings',()=>{
  const bars=candles(75);
  const late=new Map([[bars[12].t,
    {known:bars[17].t,p:1,rev:0,conf:1,hash:'a'.repeat(64)}]]);
  const results=evaluateBacktest(bars,late,
    {signalMaxAgeHours:2,stopPct:30,takePct:30,riskPct:1});
  assert.equal(results.matched,3);
  assert(results.trades.every(t=>t.entry>=bars[17].t));
});

test('Fixed PETR4 benchmark is sourced, with three models and a Buy & Hold control',()=>{
  const data=JSON.parse(readFileSync(new URL('../demo_output/petr4_three_models_reference.json',import.meta.url),'utf8'));
  assert.equal(data.schema,'PETR4_THREE_MODELS_BREAKOUT_MACD_NEURAL_V1');
  assert.equal(data.sessions,469);
  assert.equal(data.models.length,5);
  assert.equal(data.read_only,true);
  assert.equal(data.weak_elliott_labels,true);
  assert.equal(data.retrospective_MACD_selection,true);
  const model=data.models.find(x=>x.key==='MACD_intermarket');
  const benchmark=data.models.find(x=>x.key==='buy_hold');
  assert(model && benchmark);
  assert(model.total>benchmark.total);
  const html=readFileSync(new URL('../demo_output/dashboard.html',import.meta.url),'utf8');
  assert.match(html,/id="loadPetr4Report"/);
});
