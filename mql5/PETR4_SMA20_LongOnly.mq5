//+------------------------------------------------------------------+
//| PETR4_SMA20_LongOnly.mq5                                         |
//| Reproducible MQL5 analogue of PETR4 SMA20 strategy.               |
//| NOT independently compiled or broker-tested.                    |
//| Tested research used ex-post dividend-adjusted OHLC; by default |
//| MT5 uses broker's raw quotes, so results will differ unless the  |
//| broker/custom symbol supplies the same adjusted history.        |
//+------------------------------------------------------------------+
#property strict
#property version "1.00"
#property description "Long-only SMA20 on closed D1 bar, next session first tick"
#property description "Research educational EA; results need broker-specific forward verification"
#include <Trade/Trade.mqh>

input ENUM_TIMEFRAMES InpTimeframe = PERIOD_D1;
input int             InpSMAPeriod = 20;
input double          InpCapitalFraction = 0.98;
input ulong           InpMagic = 27112026;
input int             InpDeviationPoints = 20;

CTrade trade;
int maHandle = INVALID_HANDLE;
datetime lastObservedBar = 0;

int OnInit()
{
   if(InpSMAPeriod < 2 || InpCapitalFraction <= 0 || InpCapitalFraction > 1.0)
      return INIT_PARAMETERS_INCORRECT;
   maHandle = iMA(_Symbol,InpTimeframe,InpSMAPeriod,0,MODE_SMA,PRICE_CLOSE);
   if(maHandle == INVALID_HANDLE) return INIT_FAILED;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpDeviationPoints);
   trade.SetTypeFillingBySymbol(_Symbol);
   // Prevent immediate trades on attaching EA midway through a session.
   lastObservedBar = iTime(_Symbol,InpTimeframe,0);
   Print("SMA20 prepared. Validate B3 trading contract size, costs, dividends and symbol feed.");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(maHandle != INVALID_HANDLE) IndicatorRelease(maHandle);
}

double BuyVolume()
{
   double ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double contractSize = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_CONTRACT_SIZE);
   double volumeStep = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double volumeMin = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double volumeMax = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
   if(ask<=0 || contractSize<=0 || volumeStep<=0 || volumeMin<=0 || volumeMax<volumeMin)
      return 0.0;
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double budget = MathMin(equity*InpCapitalFraction,freeMargin*InpCapitalFraction);
   if(budget<=0)return 0.0;
   double requested = MathFloor(budget/(ask*contractSize*volumeStep))*volumeStep;
   double quantity = MathMin(requested,volumeMax);
   quantity = MathFloor(quantity/volumeStep)*volumeStep;
   if(quantity<volumeMin) return 0.0;
   return NormalizeDouble(quantity,8);
}

void OnTick()
{
   datetime nowBar=iTime(_Symbol,InpTimeframe,0);
   if(nowBar==0 || nowBar==lastObservedBar) return;
   if(BarsCalculated(maHandle)<InpSMAPeriod+2)return;
   double average[];
   ArrayResize(average,1);
   if(CopyBuffer(maHandle,0,1,1,average)!=1)return;
   double priorClose=iClose(_Symbol,InpTimeframe,1);
   if(priorClose<=0 || average[0]<=0)return;

   lastObservedBar=nowBar;
   bool shouldHold=priorClose>average[0];
   bool hasPosition=PositionSelect(_Symbol);
   if(hasPosition)
   {
      // Do not interfere with user/manual positions or another EA.
      if((ulong)PositionGetInteger(POSITION_MAGIC)!=InpMagic)
      {
         Print("Foreign position on ",_Symbol," — skipping signal");
         return;
      }
      if(!shouldHold)
      {
         if(!trade.PositionClose(_Symbol))
            Print("SMA20 close failure: ",trade.ResultRetcodeDescription());
      }
   }
   else if(shouldHold)
   {
      double quantity=BuyVolume();
      if(quantity<=0)
      {
         Print("SMA20: insufficient capacity or invalid volume parameters");
         return;
      }
      if(!trade.Buy(quantity,_Symbol,0.0,0.0,0.0,"SMA20 long-only"))
         Print("SMA20 entry failure: ",trade.ResultRetcodeDescription());
   }
}
