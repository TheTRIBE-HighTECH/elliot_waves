import {PHI,FIB,parseCSV,normalizeBars,normalizeGraph,graphSeries,makeClock,
  causalPivots,fibRatios,estimateHurst,normalizePredictions,evaluateBacktest,
  graphTree,pretty} from './dashboard-core.mjs';

const $=id=>document.getElementById(id);
const NS='http://www.w3.org/2000/svg';
const state={bars:[],nodes:[],graph:null,clock:null,calendar:null,source:'demonstração sintética',
  synthetic:true,asset:'BTCUSDT',predictions:null,report:null,
  annotations:[],start:0,end:0,graphVisible:[],neuralResults:null};
const chartColor={calendar:'#36d8ca',information:'#f6bb62',speed:'#ac9df9',
  neutral:'#8499b0'};
const sliders=['alpha','beta','gamma','replyWeight','depthWeight','halfLife',
  'window','visibleBars','reversal','fibTolerance','hurstWindow','maxNodes',
  'minDepth','graphHorizon','epochs','dropout','horizon','waveProb','revThr','confThr'];
const clockFields=['clockSource','alpha','beta','gamma','replyWeight','depthWeight','halfLife',
  'window','minSpeed','maxSpeed'];
const analysisFields=['visibleBars','reversal','fibTolerance','hurstWindow','cutoff',
  'showPivots','showAnnotations','topicFilter','maxNodes','minDepth','graphHorizon'];
const numeric=id=>Number($(id).value);
function notify(msg,error=false){
  const n=$('toast');n.textContent=msg;n.classList.toggle('error',error);
  n.classList.add('show');clearTimeout(notify.timer);
  notify.timer=setTimeout(()=>n.classList.remove('show'),5200);
}
function errorMessage(err){notify(err?.message||String(err),true);}
function setText(id,text){if($(id))$(id).textContent=text;}
function xdate(t){return new Date(t).toLocaleString('pt-BR',{day:'2-digit',month:'2-digit',year:'2-digit',hour:'2-digit',minute:'2-digit',timeZone:'UTC'})+' UTC';}
function readFile(file){return new Promise((resolve,reject)=>{
  const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));
  reader.onerror=()=>reject(reader.error||new Error('Erro ao ler arquivo'));reader.readAsText(file);
});}
function formatDisplay(id){
  const v=numeric(id);
  const useInt=['window','visibleBars','fibTolerance','hurstWindow','maxNodes',
    'minDepth','graphHorizon','halfLife','epochs','horizon'].includes(id);
  const text=useInt?String(Math.round(v)):pretty(v,v<1?2:1);
  setText(id+'Out',text);
}
function getClockParams(){
  const minSpeed=numeric('minSpeed'),maxSpeed=numeric('maxSpeed');
  if(!Number.isFinite(minSpeed)||!Number.isFinite(maxSpeed)||minSpeed<=0||minSpeed>=maxSpeed)
    throw Error('A velocidade mínima deve ser positiva e menor que a máxima');
  return {source:$('clockSource').value,alpha:numeric('alpha'),beta:numeric('beta'),
    gamma:numeric('gamma'),replyWeight:numeric('replyWeight'),
    depthWeight:numeric('depthWeight'),halfLife:numeric('halfLife'),
    window:numeric('window'),minSpeed,maxSpeed};
}
function getBacktestParams(){
  const cfg={capital:numeric('capital'),feeBps:numeric('feeBps'),
    slippageBps:numeric('slippageBps'),riskPct:numeric('riskPct'),
    stopPct:numeric('stopPct'),takePct:numeric('takePct'),
    threshold:numeric('waveProb'),maxReversal:numeric('revThr'),
    minConfidence:numeric('confThr')};
  for(const [k,v] of Object.entries(cfg))if(!Number.isFinite(v)||v<0)
    throw Error('Parâmetro de backtest inválido: '+k);
  if(cfg.capital<=0||cfg.stopPct<=0||cfg.riskPct<=0||cfg.takePct<=0)
    throw Error('Capital, risco, stop e take devem ser positivos');
  return cfg;
}
function currentSettings(){
  return {clock:getClockParams(),geometry:{
    visibleBars:numeric('visibleBars'),reversalPct:numeric('reversal'),
    fibonacciTolerancePct:numeric('fibTolerance'),hurstWindow:numeric('hurstWindow'),
    cutoffPercent:numeric('cutoff'),showPivots:$('showPivots').checked,
    showAnnotations:$('showAnnotations').checked},
  graph:{topic:$('topicFilter').value,maxNodes:numeric('maxNodes'),
    minDepth:numeric('minDepth'),horizonHours:numeric('graphHorizon')},
  neuralTraining:{architecture:$('architecture').value,epochs:numeric('epochs'),
    dropout:numeric('dropout'),horizonHours:numeric('horizon'),
    scales:$('scales').value.split(',').map(Number),seed:numeric('seed'),
    note:'Parâmetros para reentrenar no backend; NÃO alteram os pesos importados'},
  backtest:getBacktestParams()};
}
function loadSettings(){
  try{
    const saved=JSON.parse(localStorage.getItem('elliott-lab-settings-v1')||'null');
    if(!saved||typeof saved!=='object')return;
    const keys=['clockSource',...sliders,'minSpeed','maxSpeed','capital','feeBps',
      'slippageBps','riskPct','stopPct','takePct','cutoff','architecture','scales','seed'];
    for(const k of keys)if(Object.hasOwn(saved,k)&&$(k)){
      if($(k).tagName==='SELECT'&&![...$(k).options].some(o=>o.value===String(saved[k])))continue;
      $(k).value=String(saved[k]);
    }
  }catch(_){}
}
function saveSettings(){
  const settings={};for(const k of ['clockSource',...sliders,'minSpeed','maxSpeed',
      'capital','feeBps','slippageBps','riskPct','stopPct','takePct',
      'cutoff','architecture','scales','seed'])if($(k))settings[k]=$(k).value;
  try{localStorage.setItem('elliott-lab-settings-v1',JSON.stringify(settings));}catch(_){}
}
function downloadFile(name,content,type='text/plain'){
  const blob=new Blob([content],{type}),url=URL.createObjectURL(blob);
  const link=document.createElement('a');link.href=url;link.download=name;
  document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),500);
}
function toCSV(rows){
  if(!rows.length)return '';
  const keys=Object.keys(rows[0]),q=s=>{const t=String(s??'');return /[",\r\n]/.test(t)?'"'+t.replace(/"/g,'""')+'"':t;};
  return [keys.map(q).join(','),...rows.map(r=>keys.map(k=>q(r[k])).join(','))].join('\n');
}
function createSVG(tag,props={},text=''){
  const el=document.createElementNS(NS,tag);
  for(const [k,v] of Object.entries(props))el.setAttribute(k,String(v));
  if(text)el.textContent=text;return el;
}
function svgClear(svg){while(svg.firstChild)svg.removeChild(svg.firstChild);}
function placeholder(svg,label){
  svgClear(svg);svg.append(createSVG('text',{x:'50%',y:'50%',fill:'#91a4bd',
    'text-anchor':'middle','font-size':'12'},label));
}
function buildLine(svg,values,xvals,color,params={}){
  svgClear(svg);
  if(values.length<2)return placeholder(svg,'Dados insuficientes');
  const W=params.width||600,H=params.height||240,top=13,bottom=31,left=64,right=15;
  const xs=xvals||values.map((_,i)=>i);
  const good=values.filter(Number.isFinite);
  if(!good.length)return placeholder(svg,'Série sem valores válidos');
  let min=Math.min(...good),max=Math.max(...good);
  const pad=(max-min)*.10||Math.max(1,Math.abs(max)*.05);
  min-=pad;max+=pad;
  const a=xs[0],b=xs[xs.length-1],range=b-a||1;
  const xp=x=>left+(x-a)/range*(W-left-right),yp=y=>top+(max-y)/(max-min)*(H-top-bottom);
  for(let k=0;k<=4;k++){
    const y=top+(H-top-bottom)*k/4;
    svg.append(createSVG('line',{x1:left,x2:W-right,y1:y,y2:y,class:'chart-gridline'}));
    const lab=max-(max-min)*k/4;
    svg.append(createSVG('text',{x:left-7,y:y+3,class:'chart-axis','text-anchor':'end'},pretty(lab,params.digits??2)));
  }
  svg.append(createSVG('text',{x:left,y:H-7,class:'chart-axis'},params.startLabel||'Início'));
  svg.append(createSVG('text',{x:W-right,y:H-7,class:'chart-axis','text-anchor':'end'},params.endLabel||'Fim'));
  const points=values.map((y,j)=>Number.isFinite(y)?[xp(xs[j]),yp(y)]:null);
  let d='';
  for(const p of points){if(!p)continue;d+=(d?'L':'M')+p[0].toFixed(2)+' '+p[1].toFixed(2)+' ';}
  svg.append(createSVG('path',{d,fill:'none',stroke:color,'stroke-width':'2.3',
    'stroke-linecap':'round','stroke-linejoin':'round'}));
  if(params.pivots){
    for(const pivot of params.pivots){
      const j=pivot.at-params.offset;
      if(j<0||j>=points.length||!points[j])continue;
      const [x,y]=points[j];
      svg.append(createSVG('circle',{cx:x,cy:y,r:'3.5',fill:pivot.type==='peak'?'#f6bb62':'#36d8ca',
        stroke:'#08121f','stroke-width':'1.3'}));
    }
  }
  if(params.annotations){
    for(const point of params.annotations){
      const j=point.index-params.offset;
      if(j<0||j>=points.length||!points[j])continue;
      const [x,y]=points[j];
      svg.append(createSVG('circle',{cx:x,cy:y,r:'6',fill:'#b5a6ff',stroke:'#080f1b','stroke-width':'2'}));
      svg.append(createSVG('text',{x:x+7,y:y-9,fill:'#c5b9ff','font-size':'12','font-weight':'700'},point.wave));
    }
  }
  return {xp,yp,points,offset:params.offset,a,b,left,right,W,H};
}
function makeSynthetic(n=1600){
  let seed=73;function rand(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
  const rows=[],start=Date.UTC(2026,9,1,0,0,0);let price=66000;
  for(let i=0;i<n;i++){
    const burst=Math.exp(-(((i-420)/85)**2))+1.2*Math.exp(-(((i-1220)/92)**2));
    const volume=Math.exp(1.7+.75*burst+.3*(rand()-.5));
    const previous=price,noise=(rand()-.5)*.001;
    price=Math.max(1000,previous*Math.exp(.00012*Math.sin(i/21)+.0002*Math.sin(i/119)+noise));
    rows.push({timestamp:new Date(start+i*3600000).toISOString(),
      open:previous.toFixed(3),high:(Math.max(previous,price)*1.001).toFixed(3),
      low:(Math.min(previous,price)*.999).toFixed(3),close:price.toFixed(3),
      volume:volume.toFixed(4),num_trades:Math.round(80*volume)});
  }
  return rows;
}
async function loadDemo(){
  let rows;
  try{
    const r=await fetch('/relogio_informacional.csv',{cache:'no-store'});
    if(!r.ok)throw Error('Dataset não disponível');
    rows=parseCSV(await r.text());state.source='demo_output/relogio_informacional.csv · sintético';
  }catch(_){
    rows=makeSynthetic();state.source='série gerada localmente · sintética';
  }
  state.bars=normalizeBars(rows);state.nodes=[];state.graph=null;
  state.synthetic=true;state.predictions=null;state.annotations=[];state.neuralResults=null;
  setText('datasetTag','DEMONSTRAÇÃO SINTÉTICA');setText('modeBadge','SIMULAÇÃO · NÃO É PREVISÃO');
  $('clockSource').value='volume';refreshSources();renderAll();
  notify('Demonstração sintética carregada; importe dados reais para análise empírica.');
}
function topicList(){
  const select=$('topicFilter'),previous=select.value;
  select.replaceChildren(new Option('Todos os tópicos',''));
  const keys=[...new Set(state.nodes.map(x=>x.topic).filter(Boolean))].sort();
  for(const key of keys)select.add(new Option(key,key));
  if(keys.includes(previous))select.value=previous;
}
function refreshSources(){
  topicList();
  const graphNodes=state.nodes.length,
        graphEdges=state.graph?.validEdges||0,
        preds=state.predictions?[...state.predictions.series.calendar.keys()].length+
          [...state.predictions.series.information.keys()].length:0;
  setText('filesStatus',[
    'Mercado: '+state.source+' ('+state.bars.length+' candles)',
    'Grafo: '+graphNodes+' nós / '+graphEdges+' arestas válidas',
    'Previsões neurais: '+preds+' linhas válidas',
    'Relatório neural: '+(state.report?'importado':'não importado')
  ].join('\n'));
  setText('marketCount',state.bars.length.toLocaleString('pt-BR')+' candles');
  setText('datasetTag',state.synthetic?'DEMONSTRAÇÃO SINTÉTICA':'OHLCV REAL · IMPORTADO');
  setText('graphStatus',graphNodes?'GRAFO REAL IMPORTADO':'SEM GRAFO');
  setText('neuralStatus',preds?'PREDIÇÕES IMPORTADAS':'SEM PREVISÕES');
}
function filteredNodes(){
  const topic=$('topicFilter').value;
  return state.nodes.filter(x=>!topic||x.topic.includes(topic));
}
function recalc(){
  if(!state.bars.length)return;
  const mode=$('clockSource').value;
  if((mode==='graph'||mode==='mixed')&&!state.nodes.length){
    $('clockSource').value='volume';
    throw Error('Importe um grafo real para ativar o relógio de grafo; o modo volume foi mantido.');
  }
  state.graph=graphSeries(state.bars,filteredNodes());
  state.calendar=makeClock(state.bars,state.graph,{source:'calendar'});
  state.clock=makeClock(state.bars,state.graph,getClockParams());
  const cutoff=Math.max(30,Math.min(state.bars.length,Math.ceil(state.bars.length*numeric('cutoff')/100)));
  state.end=cutoff;state.start=Math.max(0,cutoff-numeric('visibleBars'));
  setText('cutoffLabel','Até '+xdate(state.bars[cutoff-1].t)+' ('+cutoff+' de '+state.bars.length+' candles)');
}
function showNotice(){
  const text=state.synthetic?
    'Base demonstrativa sintética. Nenhum resultado constitui evidência de eficácia no mercado.':
    'Preços importados. O grafo (se disponível) contém relações públicas reais de resposta; elas não demonstram causalidade financeira.';
  $('notice').classList.toggle('good',!state.synthetic);
  $('notice').replaceChildren(document.createTextNode('Procedência: '+text));
}
function drawMarket(){
  const bars=state.bars,end=state.end,start=state.start;
  const sub=bars.slice(start,end),prices=sub.map(b=>b.close),dates=sub.map(b=>b.t);
  const pivots=$('showPivots').checked?causalPivots(bars,numeric('reversal'),80,end):[];
  const annotations=$('showAnnotations').checked?state.annotations.filter(a=>a.index<end):[];
  const from=xdate(dates[0]),to=xdate(dates[dates.length-1]);
  const calAxis=state.calendar.axis.slice(start,end),inform=state.clock.axis.slice(start,end);
  state.lineGeometry=buildLine($('calendarChart'),prices,calAxis,chartColor.calendar,{
    offset:start,pivots,annotations,startLabel:from,endLabel:to});
  buildLine($('informationChart'),prices,inform,chartColor.information,{
    offset:start,pivots,annotations,startLabel:'τ = '+pretty(inform[0],1),endLabel:'τ = '+pretty(inform[inform.length-1],1)});
  buildLine($('speedChart'),Array.from(state.clock.speed.slice(start,end)),null,chartColor.speed,{
    height:165,digits:2,startLabel:from,endLabel:to});
  const j=end-1,prev=Math.max(0,j-24),delta=100*(bars[j].close/bars[prev].close-1);
  setText('statPrice',pretty(bars[j].close,2));
  setText('statPriceSub',(delta>=0?'+':'')+pretty(delta,2)+'% nas últimas '+(j-prev)+' barras');
  setText('statSpeed',pretty(state.clock.speed[j],3)+'×');
  setText('statSpeedSub','Relógio: '+$('clockSource').selectedOptions[0].text);
  setText('statEdges',String(state.graph.validEdges));
  setText('statEdgesSub','Nós: '+state.graph.included+' · Cobertura: '+pretty(state.graph.coverage*100,1)+'%');
  const cal=estimateHurst(bars,state.calendar.axis,end,numeric('hurstWindow'));
  const inf=estimateHurst(bars,state.clock.axis,end,numeric('hurstWindow'));
  setText('statHurst',cal?pretty(cal.H,3):'—');
  setText('statHurstSub','Estimativa em '+numeric('hurstWindow')+' barras');
  setText('hurstCalendar',cal?pretty(cal.H,3):'—');
  setText('hurstInformation',inf?pretty(inf.H,3):'—');
  setText('hurstCalendarR2',cal?'R² '+pretty(cal.r2,3):'Sem observações suficientes');
  setText('hurstInformationR2',inf?'R² '+pretty(inf.r2,3):'Sem observações suficientes');
  const ratios=fibRatios(pivots);
  const target=[.382,.5,.618,.786,1,1.618,2.618],tol=numeric('fibTolerance')/100;
  const root=$('fibDetails');root.replaceChildren();
  if(!ratios.length){root.textContent='Ainda não existem seis pivôs confirmados para avaliar as razões.';return;}
  for(let i=0;i<ratios.length;i++){
    const r=ratios[i],best=target.reduce((a,b)=>Math.abs(b-r)<Math.abs(a-r)?b:a);
    const chip=document.createElement('div');chip.className='fib-chip'+(Math.abs(r-best)/best<=tol?' match':'');
    chip.textContent=['Onda 2 / 1','Onda 3 / 1','Onda 4 / 3','Onda 5 / 1'][i]+' · '+pretty(r,3)+' (≈'+pretty(best,3)+')';
    root.append(chip);
  }
  const note=document.createElement('small');note.className='text-muted';note.textContent=
    'Tolerância '+numeric('fibTolerance')+'% · apenas comparação geométrica, sem rótulo neural.';
  root.append(note);
}
function renderGraph(){
  const nodes=filteredNodes(),end=state.end,t=state.bars[end-1].t;
  const horizon=numeric('graphHorizon')*3600000,visible=nodes.filter(x=>x.t<=t&&x.t>=t-horizon);
  const depth=numeric('minDepth'),maxNodes=numeric('maxNodes');
  const tree=graphTree(visible,t,maxNodes,$('topicFilter').value).filter(x=>x.depth>=depth);
  state.graphVisible=tree;
  const root=$('graphChart');svgClear(root);
  if(!nodes.length){placeholder(root,'Importe o arquivo CSV de respostas reais');setText('graphLegend','Sem rede social carregada.');}
  else if(!tree.length){placeholder(root,'Não há eventos no período ou filtro selecionado');setText('graphLegend','Amplie a janela do grafo ou reduza os filtros.');}
  else{
    const W=700,H=340,pad=24,maxDepth=Math.max(1,...tree.map(x=>x.depth));
    // Position by actual time, with horizontal depth perturbation for tree clarity.
    const minT=Math.min(...tree.map(x=>x.t)),maxT=Math.max(...tree.map(x=>x.t));
    const span=maxT-minT||1;
    const pos=new Map(tree.map((n,i)=>{
      const x=pad+(n.t-minT)/span*(W-pad*2);
      const y=pad+((i+.5)/tree.length)*(H-pad*2);
      return [n.id,[x,y]];
    }));
    let edges=0;
    for(const node of tree){
      const a=pos.get(node.parent),b=pos.get(node.id);
      if(!a||!b)continue;edges++;
      root.append(createSVG('path',{d:'M'+a[0]+' '+a[1]+' Q'+(a[0]+b[0])/2+' '+a[1]+' '+b[0]+' '+b[1],
        fill:'none',stroke:'#46627b','stroke-width':'1.3',opacity:'.7'}));
    }
    for(const node of tree){
      const [x,y]=pos.get(node.id);
      const circle=createSVG('circle',{cx:x,cy:y,r:node.parent?'3.8':'6.5',
        fill:node.parent?'#36d8ca':'#f6bb62',stroke:'#06101b','stroke-width':'1.1'});
      circle.append(createSVG('title',{},'ID '+node.id+' · '+node.topic+' · grau '+node.depth+' · '+xdate(node.t)));
      root.append(circle);
    }
    setText('graphLegend',tree.length+' nós mostrados · '+edges+' ligações visíveis · amostra da maior discussão no intervalo · eixo X = momento de publicação');
  }
  const stats=$('graphStats');stats.replaceChildren();
  const measures=[
    ['Nós importados',nodes.length],['Nós até o corte',nodes.filter(x=>x.t<=t).length],
    ['Respostas válidas',state.graph.validEdges],['Horas com eventos',state.graph.counts.filter(x=>x>0).length],
    ['Profundidade máx.',nodes.length?Math.max(...nodes.map(n=>n.depth)):0],
    ['Cobertura temporal',pretty(state.graph.coverage*100,2)+'%']
  ];
  for(const [label,v] of measures){
    const cell=document.createElement('div');cell.className='graph-measure';
    const name=document.createElement('span');name.textContent=label;
    const number=document.createElement('strong');number.textContent=String(v);
    cell.append(name,number);stats.append(cell);
  }
}
function neuralData(){
  const list=$('neuralOutputs');list.replaceChildren();
  if(!state.predictions){
    const p=document.createElement('p');p.className='empty-text';
    p.textContent='Importe CSV de inferência real, com clock, timestamp, known_at, weight_sha256, p_bull_3, p_reversal e confidence.';
    list.append(p);return;
  }
  const wrapper=document.createElement('div');wrapper.className='neural-box';
  for(const [key,label] of [['calendar','Cronológico'],['information','Informacional']]){
    const items=state.predictions.series[key];
    const recent=[...items.entries()].filter(([t])=>t<=state.bars[state.end-1].t)
      .sort((a,b)=>b[0]-a[0])[0];
    const cell=document.createElement('div');const small=document.createElement('small');small.textContent=label+' · '+items.size+' previsões';
    const strong=document.createElement('strong');strong.textContent=recent?pretty(recent[1].p*100,1)+'%':'—';
    const foot=document.createElement('em');foot.textContent=recent?'P(onda 3) · reversão '+pretty(recent[1].rev*100,1)+'% · confiança '+pretty(recent[1].conf*100,1)+'%':'Sem predição no intervalo';
    cell.append(small,strong,foot);wrapper.append(cell);
  }
  list.append(wrapper);
}
function buildTable(headers,rows,parent){
  const table=document.createElement('table'),thead=document.createElement('thead'),tr=document.createElement('tr');
  for(const x of headers){const th=document.createElement('th');th.textContent=x;tr.append(th);}thead.append(tr);table.append(thead);
  const body=document.createElement('tbody');
  for(const cells of rows){const tr=document.createElement('tr');
    for(const x of cells){const td=document.createElement('td');td.textContent=String(x);tr.append(td);}
    body.append(tr);
  }table.append(body);parent.append(table);
}
function renderReport(){
  const root=$('reportSummary');root.replaceChildren();
  const report=state.report;
  if(!report){const p=document.createElement('p');p.className='empty-text';p.textContent=
    'Sem relatório carregado. O código do modelo não executa pesos PyTorch no frontend estático.';root.append(p);return;}
  const rows=[];
  if(report.modes){
    for(const [mode,value] of Object.entries(report.modes))
      rows.push([mode,pretty(100*value.test?.macro_f1_labeled,2)+'%',pretty(value.backtest?.return_pct,2)+'%']);
  }else if(report.models){
    for(const [name,values] of Object.entries(report.models))for(const clock of ['calendar','information']){
      const result=values.test?.[clock];if(result)
        rows.push([name+' / '+clock,pretty(100*result.classification?.macro_f1_labeled,2)+'%',pretty(result.backtest?.return_pct,2)+'%']);
    }
  }else if(report.candidates){
    for(const [name,v] of Object.entries(report.candidates))
      rows.push([name,pretty(100*v.test?.calendar?.macro_f1_labeled,2)+'%',pretty(v.test_backtest?.calendar?.return_pct,2)+'%']);
  }
  if(rows.length)buildTable(['Modelo / relógio','F1 teste','Retorno teste'],rows,root);
  else {const p=document.createElement('p');p.className='empty-text';p.textContent='JSON reconhecido, mas sem métricas no formato esperado.';root.append(p);}
  const label=document.createElement('p');label.className='group-note';
  label.textContent='Métricas fixas da execução original; sliders da interface NÃO recalculam estes resultados. Rótulos de ondas podem ser heurísticos.';
  root.append(label);
}
function renderBacktest(resultA=null,resultB=null){
  const root=$('backtestSummary');root.replaceChildren();
  if(!resultA||!resultB){root.append(Object.assign(document.createElement('p'),{
    className:'empty-text',textContent:'Backtest não executado. É necessário CSV de previsões dos dois relógios com mesmo hash de pesos e OHLCV real com abertura.'}));
    placeholder($('equityChart'),'Aguardando previsões e execução do backtest A/B');return;}
  const rows=[
    ['Retorno líquido',pretty(resultA.returnPct,2)+'%',pretty(resultB.returnPct,2)+'%'],
    ['Drawdown máximo',pretty(resultA.maxDrawdownPct,2)+'%',pretty(resultB.maxDrawdownPct,2)+'%'],
    ['Operações completas',resultA.tradeCount,resultB.tradeCount],
    ['Acerto por operação',resultA.winRate===null?'—':pretty(resultA.winRate,1)+'%',resultB.winRate===null?'—':pretty(resultB.winRate,1)+'%'],
    ['Profit factor',pretty(resultA.profitFactor,2),pretty(resultB.profitFactor,2)],
    ['Exposição',pretty(resultA.exposurePct,2)+'%',pretty(resultB.exposurePct,2)+'%'],
    ['Previsões elegíveis',resultA.matched,resultB.matched]
  ];
  buildTable(['Métrica','Cronológico','Informacional'],rows,root);
  const info=document.createElement('p');info.className='group-note';
  info.textContent='Mesmo capital, custos e parâmetros; previsões com o mesmo hash de pesos. Stop priorizado caso stop e alvo sejam tocados no mesmo candle.';
  root.append(info);
  const svg=$('equityChart');
  // Overlay both curves using the same axis and scale.
  svgClear(svg);
  const c1=resultA.equity.map(x=>x.value),c2=resultB.equity.map(x=>x.value),n=Math.min(c1.length,c2.length),W=1060,H=200;
  const values=[...c1,...c2],min=Math.min(...values),max=Math.max(...values),padding=(max-min)*.08||1;
  function poly(v,color){
    const range=max-min+padding*2;
    const line=v.slice(0,n).map((y,j)=>{
      const x=53+j/Math.max(1,n-1)*(W-70),py=13+(max+padding-y)/range*(H-42);
      return (j?'L':'M')+x.toFixed(2)+' '+py.toFixed(2);
    }).join(' ');
    svg.append(createSVG('path',{d:line,stroke:color,'stroke-width':'2',fill:'none'}));
  }
  for(let k=0;k<4;k++)svg.append(createSVG('line',{x1:53,x2:W-17,y1:15+k*(H-42)/3,y2:15+k*(H-42)/3,class:'chart-gridline'}));
  poly(c1,chartColor.calendar);poly(c2,chartColor.information);
  svg.append(createSVG('text',{x:60,y:195,fill:chartColor.calendar,'font-size':'12'},'— Cronológico'));
  svg.append(createSVG('text',{x:230,y:195,fill:chartColor.information,'font-size':'12'},'— Informacional'));
}
function renderAnnotations(){
  const root=$('annotationsTable');root.replaceChildren();
  if(!state.annotations.length){
    const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=6;td.textContent='Sem contagens registradas.';tr.append(td);root.append(tr);return;
  }
  for(const [i,annotation] of state.annotations.entries()){
    const tr=document.createElement('tr');
    for(const value of [annotation.wave,annotation.degree,xdate(annotation.extreme_time),
      pretty(annotation.price,2),'humana · não verificada']){
      const td=document.createElement('td');td.textContent=String(value);tr.append(td);
    }
    const td=document.createElement('td'),button=document.createElement('button');
    button.textContent='Remover';button.addEventListener('click',()=>{
      state.annotations.splice(i,1);renderAnnotations();drawMarket();
    });td.append(button);tr.append(td);root.append(tr);
  }
}
function renderAll(){
  if(!state.bars.length)return;
  try{
    recalc();showNotice();drawMarket();renderGraph();neuralData();renderReport();refreshSources();
    renderAnnotations();
    if(state.neuralResults){state.neuralResults=null;renderBacktest();}
  }catch(e){errorMessage(e);}
}
function refreshGraph(){if(!state.bars.length)return;try{recalc();drawMarket();renderGraph();neuralData();}catch(e){errorMessage(e);}}
function setupAnnotations(){
  $('calendarChart').addEventListener('click',evt=>{
    if(!state.lineGeometry||!state.bars.length)return;
    const rect=$('calendarChart').getBoundingClientRect();
    const x=(evt.clientX-rect.left)/rect.width*600;
    const g=state.lineGeometry,desired=g.a+(x-g.left)/(g.W-g.left-g.right)*(g.b-g.a);
    let best=state.start,bestDist=Infinity;
    for(let i=state.start;i<state.end;i++){
      const dist=Math.abs(state.calendar.axis[i]-desired);
      if(dist<bestDist){best=i;bestDist=dist;}
    }
    const b=state.bars[best];
    state.annotations.push({
      index:best,wave:$('waveLabel').value,degree:$('waveDegree').value,
      direction:$('waveDirection').value,extreme_time:b.t,price:b.close,
      known_at:Date.now(),label_status:'human_unverified',asset:state.asset
    });
    drawMarket();renderAnnotations();
    notify('Âncora '+$('waveLabel').value+' registrada em '+xdate(b.t));
  });
  $('exportAnnotations').addEventListener('click',()=>{
    if(!state.annotations.length)return notify('Registre pelo menos uma âncora antes de exportar.',true);
    const csv=toCSV(state.annotations.map(x=>({
      asset:x.asset,wave:x.wave,degree:x.degree,direction:x.direction,
      extreme_time:new Date(x.extreme_time).toISOString(),extreme_price:x.price,
      known_at:new Date(x.known_at).toISOString(),label_status:x.label_status,
      source:'elliott_dashboard_human_annotation',annotator:'user_session'
    })));
    downloadFile('elliott_contagens_humanas.csv',csv,'text/csv;charset=utf-8');
  });
  $('clearAnnotations').addEventListener('click',()=>{
    if(!state.annotations.length)return;state.annotations=[];renderAnnotations();drawMarket();notify('Contagens locais limpas.');
  });
}
function assertMatchingWeights(pred){
  const a=pred.hashes.calendar,b=pred.hashes.information;
  if(a.size!==1||b.size!==1)throw Error('Cada relógio deve ter exatamente um hash SHA-256 de pesos.');
  const ha=[...a][0],hb=[...b][0];
  if(ha!==hb)throw Error('Os relógios usam pesos diferentes. Comparação A/B recusada.');
  return ha;
}
function startBacktest(scroll=true){
  try{
    if(state.synthetic)throw Error('Backtest neural exige histórico OHLCV real. Demonstração sintética não é aceita.');
    if(!state.predictions)throw Error('Importe previsões neurais para ambos os relógios.');
    const hash=assertMatchingWeights(state.predictions);
    const p=getBacktestParams();
    const a=evaluateBacktest(state.bars,state.predictions.series.calendar,p,state.end);
    const b=evaluateBacktest(state.bars,state.predictions.series.information,p,state.end);
    if(!a.matched||!b.matched)throw Error('Nenhuma previsão elegível para o período escolhido.');
    renderBacktest(a,b);state.neuralResults={a,b,hash};
    if(scroll)$('sec-backtest').scrollIntoView({behavior:'smooth'});
    notify('Backtest A/B executado com os mesmos pesos e custos.');
  }catch(e){errorMessage(e);}
}
async function importCSV(kind,file){
  if(!file)return;
  try{
    const rows=parseCSV(await readFile(file));
    if(kind==='market'){
      const bars=normalizeBars(rows);
      state.bars=bars;state.synthetic=false;state.source=file.name;
      state.asset=(file.name.split(/[_\-.]/)[0]||'BTCUSDT').toUpperCase();
      state.annotations=[];state.nodes=[];state.predictions=null;state.report=null;
      state.neuralResults=null;
      $('clockSource').value='volume';
      notify(bars.length+' candles de mercado importados; preços não foram fabricados.');
    }else if(kind==='graph'){
      state.nodes=normalizeGraph(rows,state.asset);
      if(!state.nodes.length)state.nodes=normalizeGraph(rows,'');
      if(!state.nodes.length)throw Error('Não encontrei nós válidos no arquivo de grafo.');
      notify(state.nodes.length+' eventos reais de grafo carregados.');
    }else if(kind==='predictions'){
      const out=normalizePredictions(rows);
      const count=out.series.calendar.size+out.series.information.size;
      if(!count)throw Error('Não encontrei previsões com clock, known_at, probabilidades e hash de pesos.');
      state.predictions=out;state.neuralResults=null;
      renderBacktest();notify(count+' previsões neurais importadas; hashes serão conferidos no backtest.');
    }
    renderAll();
  }catch(err){errorMessage(err);}
}
async function importReport(file){
  if(!file)return;
  try{state.report=JSON.parse(await readFile(file));renderReport();refreshSources();notify('Relatório neural importado. Resultados fixos não são reestimados com os sliders.');}
  catch(e){errorMessage(e);}
}
function schemaExample(){
  const text=toCSV([
    {clock:'calendar',timestamp:'2025-07-01T00:00:00Z',known_at:'2025-07-01T01:00:00Z',
      weight_sha256:'MESMOS_PESOS_HASH_DE_64_CARACTERES',p_bull_3:'.41',p_reversal:'.35',confidence:'.67'},
    {clock:'information',timestamp:'2025-07-01T00:00:00Z',known_at:'2025-07-01T01:00:00Z',
      weight_sha256:'MESMOS_PESOS_HASH_DE_64_CARACTERES',p_bull_3:'.47',p_reversal:'.30',confidence:'.71'}
  ]);
  downloadFile('formato_inferencia_neural_EXEMPLO_NAO_REAL.csv',text,'text/csv;charset=utf-8');
}
function setup(){
  loadSettings();
  for(const id of sliders)formatDisplay(id);
  for(const id of [...clockFields,...analysisFields]){
    const el=$(id);if(!el)continue;
    el.addEventListener(el.type==='range'?'input':'change',()=>{
      if(sliders.includes(id))formatDisplay(id);
      saveSettings();
      if(['topicFilter','minDepth','maxNodes','graphHorizon'].includes(id)){refreshGraph();return;}
      renderAll();
    });
  }
  for(const id of ['architecture','epochs','dropout','horizon','scales','seed',
    'waveProb','revThr','confThr','capital','riskPct','feeBps','slippageBps','stopPct','takePct']){
    const el=$(id);el.addEventListener(el.type==='range'?'input':'change',()=>{
      if(sliders.includes(id))formatDisplay(id);
      saveSettings();
      if(state.neuralResults&&['waveProb','revThr','confThr','capital','riskPct',
          'feeBps','slippageBps','stopPct','takePct'].includes(id))startBacktest(false);
    });
  }
  $('marketFile').addEventListener('change',e=>importCSV('market',e.target.files[0]));
  $('graphFile').addEventListener('change',e=>importCSV('graph',e.target.files[0]));
  $('predictionsFile').addEventListener('change',e=>importCSV('predictions',e.target.files[0]));
  $('reportFile').addEventListener('change',e=>importReport(e.target.files[0]));
  $('loadDemo').addEventListener('click',loadDemo);
  $('runBacktest').addEventListener('click',startBacktest);
  $('downloadExampleSchema').addEventListener('click',schemaExample);
  $('exportConfig').addEventListener('click',()=>{
    try{
      const json=JSON.stringify({schema:'elliott-dashboard-1.0',exported_utc:new Date().toISOString(),
        market:{source:state.source,synthetic:state.synthetic,asset:state.asset,candles:state.bars.length},
        params:currentSettings()},null,2);
      downloadFile('elliott_experiment_config.json',json,'application/json');
      notify('Configuração de pesquisa exportada; parâmetros neurais exigem reexecução no backend.');
    }catch(e){errorMessage(e);}
  });
  setupAnnotations();
  renderBacktest();
  fetch('/dashboard_reference.json').then(r=>r.ok?r.json():null).then(data=>{
    if(data&&!state.report){state.report=data;renderReport();}
  }).catch(()=>{});
  loadDemo();
}
setup();
