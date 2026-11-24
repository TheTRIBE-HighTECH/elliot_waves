// Pure, dependency-free analytics shared by browser and node tests.
export const PHI=(1+Math.sqrt(5))/2;
export const FIB=[1,2,3,5,8,13,21,34,55,89,144];
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const isNum=x=>Number.isFinite(x);
export function parseCSV(text){
  const table=[];let row=[],field='',quoted=false;
  text=String(text).replace(/^\uFEFF/,'');
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){
      if(quoted&&text[i+1]==='"'){field+='"';i++;}
      else quoted=!quoted;
    }else if(c===','&&!quoted){row.push(field);field='';}
    else if((c==='\r'||c==='\n')&&!quoted){
      if(c==='\r'&&text[i+1]==='\n')i++;
      row.push(field);field='';if(row.some(x=>x.trim()))table.push(row);row=[];
    }else field+=c;
  }
  if(quoted)throw Error('CSV: aspas não fechadas');
  row.push(field);if(row.some(x=>x.trim()))table.push(row);
  if(!table.length)return [];
  const keys=table.shift().map(x=>x.trim().toLowerCase());
  return table.map(row=>Object.fromEntries(keys.map((key,j)=>[key,(row[j]??'').trim()])));
}
export function timestamp(value){
  if(value==null||String(value).trim()==='')return NaN;
  const s=String(value).trim();
  if(/^\d{10,17}$/.test(s)){
    const n=Number(s);return n>1e14?n/1000:n>1e11?n:n*1000;
  }
  if(/^\d{2}-\d{2}-\d{4}/.test(s)){
    const m=s.match(/^(\d\d)-(\d\d)-(\d{4})(?:[ T](\d\d):(\d\d))?/);
    if(m)return Date.UTC(+m[3],+m[2]-1,+m[1],+(m[4]||0),+(m[5]||0));
  }
  return Date.parse(s);
}
function val(o,...keys){for(const key of keys)if(o[key]!==undefined&&o[key]!=='')return o[key];return undefined;}
export function normalizeBars(rows){
  const candles=[];
  for(const obj of rows){
    const t=timestamp(val(obj,'timestamp','date','open_time','time'));
    const close=Number(val(obj,'close','price'));
    const hasOpen=val(obj,'open')!==undefined;
    const open=hasOpen?Number(obj.open):close;
    const high=val(obj,'high')!==undefined?Number(obj.high):Math.max(open,close);
    const low=val(obj,'low')!==undefined?Number(obj.low):Math.min(open,close);
    const volume=Number(val(obj,'volume','quote_volume')??0);
    const trades=Number(val(obj,'num_trades','number_of_trades')??0);
    const buy=Number(val(obj,'taker_buy_volume')??0);
    if(![t,close,open,high,low,volume,trades,buy].every(isNum)||close<=0||open<=0||high<Math.max(open,close)||low>Math.min(open,close)||low<=0||volume<0)continue;
    candles.push({t,open,high,low,close,volume,trades,buy,hasOpen});
  }
  candles.sort((a,b)=>a.t-b.t);
  for(let i=1;i<candles.length;i++)if(candles[i].t===candles[i-1].t)throw Error('Candles com timestamps duplicados');
  if(candles.length<30)throw Error('Precisamos de pelo menos 30 candles válidos');
  return candles;
}
export function normalizeGraph(rows,asset=''){
  const result=[];const seen=new Set();
  for(const obj of rows){
    const a=String(val(obj,'asset')??'');
    if(asset&&a&&a!==asset)continue;
    const id=String(val(obj,'node_id','id')??'');
    const parent=String(val(obj,'parent_id')??'');
    const t=timestamp(val(obj,'created_at','time_utc','created_at_i','time'));
    const story=String(val(obj,'story_id')??id);
    const depth=Number(val(obj,'depth')??0);
    if(!id||!isNum(t)||!isNum(depth)||depth<0)continue;
    if(seen.has(id))continue;
    seen.add(id);
    result.push({id,parent,story,depth,t,topic:String(val(obj,'topic')??''),
                 kind:String(val(obj,'node_type','kind')??(parent?'reply':'story')),
                 url:String(val(obj,'source_url')??'')});
  }
  result.sort((a,b)=>a.t-b.t||a.id.localeCompare(b.id));
  return result;
}
function lowerBound(array,t){
  let lo=0,hi=array.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(array[mid]<t)lo=mid+1;else hi=mid;}
  return lo;
}
export function graphSeries(bars,nodes){
  const n=bars.length,ts=bars.map(b=>b.t),hour=bars.length>1?Math.max(1,ts[1]-ts[0]):3600000;
  const counts=new Float64Array(n),roots=new Float64Array(n),replies=new Float64Array(n),
        depth=new Float64Array(n),rapid=new Float64Array(n),connected=new Float64Array(n);
  const byId=new Map(nodes.map(x=>[x.id,x]));
  let included=0,validEdges=0;
  for(const node of nodes){
    const i=lowerBound(ts,node.t+1)-1; // event belongs to bar whose open <= event time
    if(i<0||i>=n||node.t>=ts[i]+(i+1<n?ts[i+1]-ts[i]:hour))continue;
    included++;counts[i]++;depth[i]+=node.depth;
    if(!node.parent)roots[i]++;else replies[i]++;
    const parent=byId.get(node.parent);
    if(parent&&parent.t<=node.t){
      validEdges++;connected[i]++;
      if(node.t-parent.t<=3600000)rapid[i]++;
    }
  }
  return {counts,roots,replies,depth,rapid,connected,included,validEdges,
          coverage:counts.filter(x=>x>0).length/n};
}
function rollingZ(x,i,window){
  const lo=Math.max(0,i-window);const count=i-lo;
  if(count<24)return 0;
  let sum=0,sq=0;
  for(let j=lo;j<i;j++){sum+=x[j];sq+=x[j]*x[j];}
  const mean=sum/count,sd=Math.sqrt(Math.max(1e-5,sq/count-mean*mean));
  return clamp((x[i]-mean)/Math.max(.25,sd),-3,3);
}
export function makeClock(bars,graph={},settings={}){
  const n=bars.length,params={source:'volume',alpha:.65,beta:.4,gamma:.15,
      halfLife:8,depthWeight:.2,replyWeight:1,window:168,minSpeed:.35,maxSpeed:3.5,
      ...settings};
  const volume=new Float64Array(n),depth=new Float64Array(n),graphRaw=new Float64Array(n);
  let carry=0;const decay=Math.exp(-Math.log(2)/Math.max(.5,params.halfLife));
  for(let i=0;i<n;i++){
    volume[i]=Math.log1p(Math.max(0,bars[i].volume))+0.25*Math.log1p(Math.max(0,bars[i].trades));
    const c=graph.counts?.[i]||0,rep=graph.replies?.[i]||0,dp=graph.depth?.[i]||0,
          edge=graph.connected?.[i]||0;
    const raw=Math.log1p(c+params.replyWeight*rep+params.depthWeight*dp+params.gamma*edge);
    carry=carry*decay+raw;graphRaw[i]=carry;
    depth[i]=rep>0?dp/rep:0;
  }
  const clock=new Float64Array(n),speed=new Float64Array(n),interactions=new Float64Array(n);
  for(let i=0;i<n;i++){
    const zv=rollingZ(volume,i,params.window),zg=rollingZ(graphRaw,i,params.window);
    let score=params.source==='calendar'?0:params.source==='volume'?params.alpha*zv:
       params.source==='graph'?params.alpha*zg:
       params.alpha*(params.beta*zv+(1-params.beta)*zg);
    speed[i]=params.source==='calendar'?1:clamp(Math.exp(clamp(score,-4,4)),params.minSpeed,params.maxSpeed);
    interactions[i]=graphRaw[i];
    if(i)clock[i]=clock[i-1]+speed[i]*(bars[i].t-bars[i-1].t)/3600000;
  }
  return {axis:clock,speed,interactions,params};
}
export function causalPivots(bars,reversalPercent=.7,limit=80,end=bars.length){
  const r=Math.log1p(Math.max(.05,reversalPercent)/100),p=bars.map(b=>Math.log(b.close));
  const out=[];let ref=p[0],ext=0,dir=0;
  for(let i=1;i<end;i++){
    if(dir===0){
      if(p[i]-ref>=r){dir=1;ext=i;ref=p[i];}
      else if(ref-p[i]>=r){dir=-1;ext=i;ref=p[i];}
    }else if(dir>0){
      if(p[i]>=ref){ref=p[i];ext=i;}
      else if(ref-p[i]>=r){
        out.push({at:ext,confirmed:i,price:bars[ext].close,type:'peak'});
        dir=-1;ref=p[i];ext=i;
      }
    }else{
      if(p[i]<=ref){ref=p[i];ext=i;}
      else if(p[i]-ref>=r){
        out.push({at:ext,confirmed:i,price:bars[ext].close,type:'trough'});
        dir=1;ref=p[i];ext=i;
      }
    }
  }
  return out.slice(-limit);
}
export function fibRatios(pivots){
  if(pivots.length<6)return [];
  const p=pivots.slice(-6).map(x=>x.price);
  const l=p.slice(1).map((x,i)=>Math.abs(x-p[i]));
  return [l[1]/l[0],l[2]/l[0],l[3]/l[2],l[4]/l[0]].filter(isNum);
}
export function estimateHurst(bars,axis,until=bars.length,window=400,lags=[1,2,3,5,8,13,21,34]){
  const left=Math.max(0,until-window),px=bars.map(x=>Math.log(x.close));
  const t=axis||bars.map((_,i)=>i);
  const samples=[];
  for(const lag of lags){
    const inc=[];
    for(let i=left+lag;i<until;i++){
      const target=t[i]-lag;
      if(target<t[left])continue;
      const j=lowerBound(t,target+1e-9)-1;
      if(j<left||j>=i)continue;
      const v=j+1<i?px[j]+(px[j+1]-px[j])*(target-t[j])/(t[j+1]-t[j]):px[j];
      inc.push(px[i]-v);
    }
    if(inc.length<25)continue;
    const mean=inc.reduce((a,b)=>a+b,0)/inc.length;
    const variance=inc.reduce((a,b)=>a+(b-mean)**2,0)/inc.length;
    if(variance>0)samples.push([Math.log(lag),Math.log(variance)]);
  }
  if(samples.length<3)return null;
  const mx=samples.reduce((a,b)=>a+b[0],0)/samples.length;
  const my=samples.reduce((a,b)=>a+b[1],0)/samples.length;
  const den=samples.reduce((a,b)=>a+(b[0]-mx)**2,0);
  const slope=samples.reduce((a,b)=>a+(b[0]-mx)*(b[1]-my),0)/den;
  const tss=samples.reduce((a,b)=>a+(b[1]-my)**2,0);
  const rss=samples.reduce((a,b)=>a+(b[1]-my-slope*(b[0]-mx))**2,0);
  return {H:slope/2,r2:tss>0?1-rss/tss:0,samples:samples.length};
}
export function normalizePredictions(rows){
  const out={calendar:new Map(),information:new Map()};const hashes={calendar:new Set(),information:new Set()};
  for(const row of rows){
    const mode=String(val(row,'clock','mode')??'').toLowerCase();
    const key=mode==='calendar'||mode==='cronologico'?'calendar':
      mode==='information'||mode==='informational'||mode==='graph'||mode==='informacional'?'information':null;
    if(!key)continue;
    const t=timestamp(val(row,'timestamp'));
    const known=timestamp(val(row,'known_at','available_at'));
    const p=Number(val(row,'p_bull_3','p_wave3')??NaN),
          rev=Number(val(row,'p_reversal')??NaN),
          conf=Number(val(row,'confidence')??NaN);
    const hash=String(val(row,'weight_sha256','weights_sha256')??'');
    if(!isNum(t)||!isNum(known)||![p,rev,conf].every(v=>isNum(v)&&v>=0&&v<=1)||hash.length<8)continue;
    if(known<t)continue; // deterministic close or later, no retroactive signals
    out[key].set(t,{known,p,rev,conf,hash});
    hashes[key].add(hash);
  }
  return {series:out,hashes};
}
export function evaluateBacktest(bars,predictions,params={},end=bars.length){
  const cfg={capital:10000,feeBps:10,slippageBps:3,threshold:.45,maxReversal:.50,
    minConfidence:.55,stopPct:2,takePct:4,riskPct:1,...params};
  if(!bars.every(x=>x.hasOpen))throw Error('Backtest exige preços de abertura reais (open)');
  const map=predictions instanceof Map?predictions:new Map();
  let cash=cfg.capital,position=null,peak=cash,maxDrawdown=0,exposure=0;
  const trades=[],equity=[],fee=(cfg.feeBps+cfg.slippageBps)/10000;
  let signalCount=0,matched=0,positive=0,negative=0;
  for(let i=1;i<end;i++){
    const bar=bars[i],prior=bars[i-1],pred=map.get(prior.t);
    const eligible=pred&&pred.known<=bar.t;
    const buy=Boolean(eligible&&pred.p>=cfg.threshold&&pred.rev<=cfg.maxReversal&&pred.conf>=cfg.minConfidence);
    if(eligible){matched++;if(buy)signalCount++;}
    const fillEntry=bar.open*(1+fee),fillExit=bar.open*(1-fee);
    if(position&&!buy){
      const proceeds=position.qty*fillExit;
      const pnl=proceeds-position.entryCost;cash+=proceeds;
      trades.push({entry:position.entryTime,exit:bar.t,pnl,returnPct:100*pnl/position.entryCost,reason:'signal'});
      positive+=Math.max(0,pnl);negative+=Math.max(0,-pnl);position=null;
    }else if(!position&&buy){
      const qty=Math.min(cash/(fillEntry),cash*cfg.riskPct/100/(fillEntry*cfg.stopPct/100));
      if(qty>0&&isNum(qty)){
        position={qty,entryPrice:fillEntry,entryCost:qty*fillEntry,entryTime:bar.t};
        cash-=position.entryCost;
      }
    }
    if(position){
      // Conservative stop-first priority if both stop and target touch same bar.
      const stop=position.entryPrice*(1-cfg.stopPct/100),take=position.entryPrice*(1+cfg.takePct/100);
      let exitPrice=null,reason='';
      if(bar.low<=stop){exitPrice=Math.min(stop,bar.open)*(1-fee);reason='stop';}
      else if(bar.high>=take){exitPrice=take*(1-fee);reason='take';}
      if(exitPrice!==null){
        const proceeds=position.qty*exitPrice,pnl=proceeds-position.entryCost;
        cash+=proceeds;trades.push({entry:position.entryTime,exit:bar.t,pnl,returnPct:100*pnl/position.entryCost,reason});
        positive+=Math.max(0,pnl);negative+=Math.max(0,-pnl);position=null;
      }
    }
    const total=cash+(position?position.qty*bar.close:0);
    if(position)exposure++;
    peak=Math.max(peak,total);maxDrawdown=Math.max(maxDrawdown,1-total/peak);
    equity.push({t:bar.t,value:total});
  }
  if(position){
    const bar=bars[end-1],proceeds=position.qty*bar.close*(1-fee);
    const pnl=proceeds-position.entryCost;cash+=proceeds;
    trades.push({entry:position.entryTime,exit:bar.t,pnl,returnPct:100*pnl/position.entryCost,reason:'end'});
    positive+=Math.max(0,pnl);negative+=Math.max(0,-pnl);
    if(equity.length)equity[equity.length-1].value=cash;
  }
  const wins=trades.filter(x=>x.pnl>0).length;
  return {endEquity:cash,returnPct:100*(cash/cfg.capital-1),maxDrawdownPct:100*maxDrawdown,
    trades,tradeCount:trades.length,matched,signalCount,
    winRate:trades.length?100*wins/trades.length:null,
    profitFactor:negative?positive/negative:null,
    exposurePct:100*exposure/Math.max(1,end-1),equity};
}
export function graphTree(nodes,cutoff=Infinity,limit=100,topic=''){
  const selected=nodes.filter(x=>x.t<=cutoff&&(!topic||x.topic.includes(topic)));
  const count=new Map();
  for(const n of selected)count.set(n.story,(count.get(n.story)||0)+1);
  const best=[...count.entries()].sort((a,b)=>b[1]-a[1])[0]?.[0];
  if(!best)return [];
  return selected.filter(x=>x.story===best).slice(0,limit);
}
export function pretty(num,digits=2){
  return isNum(num)?Number(num).toLocaleString('pt-BR',{maximumFractionDigits:digits,minimumFractionDigits:digits}):'—';
}
