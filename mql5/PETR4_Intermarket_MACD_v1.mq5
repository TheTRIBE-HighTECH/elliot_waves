//+------------------------------------------------------------------+
//| PETR4_Intermarket_MACD_v1.mq5                                    |
//| MQL5 standard library CTrade; observational backtest research.    |
//| IMPORTANT: UNCOMPILED; brokers' symbols, time zones, dividends   |
//| and data completeness must be validated before any live use.      |
//+------------------------------------------------------------------+
#property strict
#property version "1.00"
#property description "PETR4 MACD EMA(12,35,5) + lagged IBOV/BrentBRL trend or BrentBRL corr"
#property description "Derived from retrospective multi-market study; no proven forward edge"
#include <Trade/Trade.mqh>

enum ENUM_INTERMARKET_FILTER
  {
   FILTER_IBOV_OR_BRENT_SMA50=0,
   FILTER_BRENT_MOM_OR_CORRELATION=1
  };

input string InpPETR4Symbol = "PETR4";
input string InpBrentSymbol = "BRENT";
input string InpUSDBRLSymbol = "USDBRL";
input string InpIbovSymbol = "IBOV";
input ENUM_INTERMARKET_FILTER InpFilter = FILTER_IBOV_OR_BRENT_SMA50;
input int InpFastEMA = 12;
input int InpSlowEMA = 35;
input int InpSignalEMA = 5;
input int InpExternalSMA = 50;
input int InpBrentMomentum = 20;
input int InpCorrelationWindow = 63;
input double InpCorrelationThreshold = 0.25;
input double InpEquityFraction = 0.98;
input ulong InpMagic = 20261128;
input int InpMaxDeviationPoints = 20;

CTrade order;
datetime previousObservedDay=0;

// A closed factor bar is retrieved strictly prior to the opening timestamp
// of the selected PETR4 session. For an entry at session t open, signal
// is computed at close t-1 and market factors come from t-2 or earlier.
// Cross-market B3/forex/futures trading calendars need broker reconciliation.
bool PreviousMarketClose(const string symbol,const int petrShift,double &value)
  {
   datetime reference=iTime(InpPETR4Symbol,PERIOD_D1,petrShift);
   if(reference<=0)return false;
   int sh=iBarShift(symbol,PERIOD_D1,reference,false);
   if(sh<1)return false;
   datetime barTime=iTime(symbol,PERIOD_D1,sh);
   if(barTime<=0 || barTime>=reference)return false;
   double price=iClose(symbol,PERIOD_D1,sh);
   if(price<=0 || !MathIsValidNumber(price))return false;
   value=price;
   return true;
  }

bool BrentBRL(const int petrShift,double &out)
  {
   double oil=0,usdbrl=0;
   if(!PreviousMarketClose(InpBrentSymbol,petrShift,oil))return false;
   if(!PreviousMarketClose(InpUSDBRLSymbol,petrShift,usdbrl))return false;
   out=oil*usdbrl;
   return (out>0 && MathIsValidNumber(out));
  }

// EMA MACD with EMA SIGNAL: same definition as Python pandas
// ewm(span=n,adjust=False), rather than MetaTrader standard iMACD
// which conventionally smooths its signal line with SMA.
bool EmaMacdLong(bool &longBias)
  {
   int warmup=MathMax(320,InpSlowEMA*12);
   double prices[];
   ArraySetAsSeries(prices,false);
   int copied=CopyClose(InpPETR4Symbol,PERIOD_D1,1,warmup,prices);
   if(copied<warmup-4)return false;
   double ef=prices[0],es=prices[0],signal=0;
   double af=2.0/(InpFastEMA+1),as=2.0/(InpSlowEMA+1);
   double ag=2.0/(InpSignalEMA+1),macd=0;
   for(int i=1;i<copied;i++)
     {
      ef=af*prices[i]+(1-af)*ef;
      es=as*prices[i]+(1-as)*es;
      macd=ef-es;
      signal=ag*macd+(1-ag)*signal;
     }
   longBias=(macd>signal);
   return true;
  }

bool IbovOrBrentTrend(bool &allow)
  {
   double latestIbov=0,latestOil=0;
   if(!PreviousMarketClose(InpIbovSymbol,1,latestIbov))return false;
   if(!BrentBRL(1,latestOil))return false;
   double sumIbov=0,sumOil=0;
   for(int i=1;i<=InpExternalSMA;i++)
     {
      double ibov=0,oil=0;
      if(!PreviousMarketClose(InpIbovSymbol,i,ibov))return false;
      if(!BrentBRL(i,oil))return false;
      sumIbov+=ibov;sumOil+=oil;
     }
   allow=(latestIbov>sumIbov/InpExternalSMA) ||
         (latestOil>sumOil/InpExternalSMA);
   return true;
  }

bool PearsonBrentPETR4(double &corr)
  {
   if(InpCorrelationWindow<10)return false;
   double xsum=0,ysum=0,x2=0,y2=0,xy=0;
   for(int j=0;j<InpCorrelationWindow;j++)
     {
      // PETR4 close-to-close return ending at t-2 when PETR4 opens t.
      double p1=iClose(InpPETR4Symbol,PERIOD_D1,j+2);
      double p0=iClose(InpPETR4Symbol,PERIOD_D1,j+3);
      double b1=0,b0=0;
      // Factor available strictly before preceding PETR4 session t-1:
      // j=0: market data ending no later than t-2.
      if(p1<=0 || p0<=0 ||
         !BrentBRL(j+1,b1) || !BrentBRL(j+2,b0))return false;
      double pr=MathLog(p1/p0);
      double br=MathLog(b1/b0);
      xsum+=pr;ysum+=br;x2+=pr*pr;y2+=br*br;xy+=pr*br;
     }
   double n=(double)InpCorrelationWindow;
   double den=(n*x2-xsum*xsum)*(n*y2-ysum*ysum);
   if(den<=1e-16){corr=0;return true;}
   corr=(n*xy-xsum*ysum)/MathSqrt(den);
   return true;
  }

bool BrentMomentumOrCorrelation(bool &allow)
  {
   double bNow=0,bPast=0,corr=0;
   if(!BrentBRL(1,bNow)||!BrentBRL(InpBrentMomentum+1,bPast))return false;
   if(!PearsonBrentPETR4(corr))return false;
   allow=(bNow>bPast)||(corr>InpCorrelationThreshold);
   return true;
  }

double OrderVolume()
  {
   MqlTick tick;
   if(!SymbolInfoTick(InpPETR4Symbol,tick) || tick.ask<=0)return 0;
   double step=SymbolInfoDouble(InpPETR4Symbol,SYMBOL_VOLUME_STEP);
   double smallest=SymbolInfoDouble(InpPETR4Symbol,SYMBOL_VOLUME_MIN);
   double largest=SymbolInfoDouble(InpPETR4Symbol,SYMBOL_VOLUME_MAX);
   double contract=SymbolInfoDouble(InpPETR4Symbol,SYMBOL_TRADE_CONTRACT_SIZE);
   if(step<=0||smallest<=0||largest<smallest||contract<=0)return 0;
   double budget=AccountInfoDouble(ACCOUNT_EQUITY)*InpEquityFraction;
   double wanted=MathFloor(budget/(tick.ask*contract*step))*step;
   wanted=MathFloor(MathMin(wanted,largest)/step)*step;
   if(wanted<smallest)return 0;
   double margin=0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY,InpPETR4Symbol,wanted,tick.ask,margin))
      return 0;
   if(margin>AccountInfoDouble(ACCOUNT_MARGIN_FREE)*0.98)return 0;
   return NormalizeDouble(wanted,8);
  }

int OnInit()
  {
   if(_Symbol!=InpPETR4Symbol)return INIT_PARAMETERS_INCORRECT;
   if(InpFastEMA<=1||InpSignalEMA<=1||InpSlowEMA<=InpFastEMA ||
      InpExternalSMA<10 || InpBrentMomentum<2 ||
      InpCorrelationWindow<10 || InpEquityFraction<=0 ||
      InpEquityFraction>1)return INIT_PARAMETERS_INCORRECT;
   if(!SymbolSelect(InpBrentSymbol,true) ||
      !SymbolSelect(InpUSDBRLSymbol,true) ||
      !SymbolSelect(InpIbovSymbol,true))
      return INIT_FAILED;
   order.SetExpertMagicNumber(InpMagic);
   order.SetDeviationInPoints(InpMaxDeviationPoints);
   order.SetTypeFillingBySymbol(InpPETR4Symbol);
   previousObservedDay=iTime(InpPETR4Symbol,PERIOD_D1,0);
   Print("EA initialized; uncompiled research example. Source-adjusted historical backtest may not match broker's raw-price/live quotes.");
   return INIT_SUCCEEDED;
  }

void OnTick()
  {
   datetime thisDay=iTime(InpPETR4Symbol,PERIOD_D1,0);
   if(thisDay==0||thisDay==previousObservedDay)return;
   // Keep trying on the first tick(s) until all history has loaded.
   bool macdLong=false,marketOK=false;
   if(!EmaMacdLong(macdLong))return;
   if(InpFilter==FILTER_IBOV_OR_BRENT_SMA50)
     {
      if(!IbovOrBrentTrend(marketOK))return;
     }
   else
     {
      if(!BrentMomentumOrCorrelation(marketOK))return;
     }
   previousObservedDay=thisDay;
   bool want=macdLong&&marketOK;
   bool existing=PositionSelect(InpPETR4Symbol);
   if(existing)
     {
      if((ulong)PositionGetInteger(POSITION_MAGIC)!=InpMagic)
        {
         Print("Non-EA position exists; skipping PETR4.");
         return;
        }
      if(!want)
        {
         bool ok=order.PositionClose(InpPETR4Symbol);
         if(!ok)Print("PositionClose failed: ",order.ResultRetcodeDescription());
        }
     }
   else if(want)
     {
      double vol=OrderVolume();
      if(vol<=0)
        {
         Print("No valid order volume or free margin");
         return;
        }
      bool ok=order.Buy(vol,InpPETR4Symbol,0,0,0,"PETR4 intermarket MACD");
      if(!ok)Print("Buy failed: ",order.ResultRetcodeDescription());
     }
  }
//+------------------------------------------------------------------+
