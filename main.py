import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class OptimizedBacktestEngine:
    def __init__(self, start_date, end_date, initial_capital=50000):
        # 时间参数
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # 资金管理
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        
        # 数据缓存
        self.hist_data_cache = {}
        self.price_cache = defaultdict(dict)
        self.symbol_info_cache = {}
        
        # 持仓管理（使用DataFrame提升效率）
        self.positions = pd.DataFrame(columns=['symbol', 'shares', 'buy_price', 'buy_date'])
        self.trade_log = []
        self.daily_values = []
        
        # 并行计算参数
        self.max_workers = 8
        self.debug = False

    def initialize_engine(self):
        """初始化引擎，预加载所有必要数据"""
        # 获取股票池
        stock_pool = self._prepare_stock_pool()
        
        # 预加载历史数据
        self._preload_historical_data(stock_pool)
        
        # 预计算价格缓存
        self._precalculate_daily_prices()
        
        # 预计算技术指标
        self._precalculate_technical_indicators()

    def _prepare_stock_pool(self):
        """优化后的股票池准备"""
        # 获取全市场股票并过滤
        sh_stocks = ak.stock_sh_a_spot_em()[['代码', '名称', '成交量']]
        sz_stocks = ak.stock_sz_a_spot_em()[['代码', '名称', '成交量']]
        bj_stocks = ak.stock_bj_a_spot_em()[['代码', '名称', '成交量']]
        
        all_stocks = pd.concat([sh_stocks, sz_stocks, bj_stocks])
        
        # 过滤条件
        filtered = all_stocks[
            (~all_stocks['名称'].str.contains('ST')) &
            (~all_stocks['代码'].astype(str).str.startswith(('3', '688')))
        ]
        
        # 按成交量排序取前500
        return filtered.sort_values('成交量', ascending=False).head(5000)

    def _preload_historical_data(self, stock_pool):
        """多线程预加载历史数据"""
        symbols = []
        for _, row in stock_pool.iterrows():
            exchange = self._get_exchange(row['代码'])
            if exchange != 'UNKNOWN':
                symbols.append(f"{exchange}{row['代码']}")

        def load_data(symbol):
            try:
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    adjust="hfq",
                    start_date=self.start_date - timedelta(days=10),
                    end_date=self.end_date
                )
                df['date'] = pd.to_datetime(df['date'])
                return symbol, df
            except Exception as e:
                if self.debug:
                    print(f"预加载 {symbol} 失败: {str(e)}")
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(load_data, symbols))

        for result in results:
            if result is not None:
                symbol, df = result
                self.hist_data_cache[symbol] = df
                # 缓存最后三天数据用于三连板判断
                self.symbol_info_cache[symbol] = df[['date', 'close']].tail(3).values

    def _precalculate_daily_prices(self):
        """预计算每日价格"""
        for symbol, df in self.hist_data_cache.items():
            for _, row in df.iterrows():
                self.price_cache[row['date'].date()][symbol] = row['close']

    def _precalculate_technical_indicators(self):
        """预计算技术指标"""
        for symbol, df in self.hist_data_cache.items():
            # 计算换手率
            if 'turnover_rate' not in df.columns:
                df['turnover_rate'] = (df['amount'] / df['outstanding_share']) * 100
                
            # 计算均线指标
            df['turnover_ma5'] = df['turnover_rate'].rolling(5).mean()
            df['volume_ma5'] = df['volume'].rolling(5).mean()
            
            # 更新缓存
            self.hist_data_cache[symbol] = df

    def _get_exchange(self, code):
        """判断交易所"""
        code_str = str(code)
        if code_str.startswith(('6', '900')):
            return 'sh'
        elif code_str.startswith(('0', '3')):
            return 'sz'
        elif code_str.startswith(('4', '8')):
            return 'bj'
        return 'UNKNOWN'

    def _is_3_consecutive_limit_up(self, symbol):
        """判断最后三天是否三连板"""
        data = self.symbol_info_cache.get(symbol, [])
        if len(data) < 3:
            return False
        closes = [item[1] for item in data[-3:]]
        return all(c >= prev * 1.099 for prev, c in zip(closes[:-1], closes[1:]))

    def run_backtest(self):
        """主回测逻辑"""
        trading_dates = self._get_trading_dates()
        
        for date in trading_dates:
            if self.debug:
                print(f"\n处理日期: {date.date()}")
                
            # 卖出逻辑
            self._process_sell(date)
            print("handle sell over:")
            # 买入逻辑
            if date.weekday() < 5:
                self._process_buy(date)
            print("handle buy over:")
            # 记录净值
            self._record_daily_value(date)
            print("handle run backtest:")

    def _get_trading_dates(self):
        """生成交易日序列"""
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        return [date for date in dates if date.date() in self.price_cache]

    def _process_sell(self, date):
        """向量化卖出处理"""
        if self.positions.empty:
            return

        # 获取当日价格
        date_str = date.date()
        positions = self.positions.copy()
        positions['current_price'] = positions['symbol'].map(self.price_cache[date_str])
        positions['pct_change'] = (positions['current_price'] - positions['buy_price']) / positions['buy_price']

        # 筛选需要卖出的仓位
        sell_positions = positions[np.abs(positions['pct_change']) >= 0.1]
        
        for _, pos in sell_positions.iterrows():
            print(pos['symbol'])
            sell_shares = int(pos['shares'] * 0.3)
            if sell_shares <= 0:
                continue

            # 更新持仓
            self.positions.loc[self.positions['symbol'] == pos['symbol'], 'shares'] -= sell_shares
            self.current_cash += sell_shares * pos['current_price'] * 0.999

            # 记录交易
            self.trade_log.append({
                'date': date,
                'symbol': pos['symbol'],
                'type': 'sell',
                'shares': sell_shares,
                'price': pos['current_price']
            })

        # 清理零持仓
        self.positions = self.positions[self.positions['shares'] > 0]

    def _process_buy(self, date):
        """改进后的非并行买入处理"""
        if self.debug:
            print(f"\n=== 开始处理买入日期：{date.date()} ===")
        
        candidates = []
        date_str = date.date()
        processed_count = 0
        
        # 获取所有候选股票代码
        symbols = list(self.hist_data_cache.keys())
        
        # 调试信息
        if self.debug:
            print(f"待处理股票总数: {len(symbols)}")
            print("开始逐个检查候选股...")
        
        for symbol in symbols:
            try:
                processed_count += 1
                
                # 调试进度显示
                if self.debug and processed_count % 100 == 0:
                    print(f"已处理 {processed_count}/{len(symbols)} 只股票...")
                
                # 跳过无历史数据的股票
                df = self.hist_data_cache.get(symbol)
                if df is None or df.empty:
                    continue
                    
                # 检查数据是否包含所需日期
                if date not in df['date'].values:
                    continue
                    
                # 三连板检查
                if not self._is_3_consecutive_limit_up(symbol):
                    continue
                    
                # 获取最新数据
                last_day = df[df['date'] == date].iloc[-1]
                
                # 字段有效性检查
                required_fields = ['turnover_rate', 'turnover_ma5', 'volume', 'volume_ma5']
                if any(pd.isna(last_day[field]) for field in required_fields):
                    continue
                    
                # 条件判断
                cond1 = last_day['turnover_rate'] > 1.5 * last_day['turnover_ma5']
                cond2 = last_day['volume'] > 2 * last_day['volume_ma5']
                
                if cond1 and cond2:
                    candidates.append({
                        'symbol': symbol,
                        'price': last_day['close'],
                        'turnover': last_day['turnover_rate'],
                        'volume': last_day['volume']
                    })
                    
                    if self.debug:
                        print(f"发现候选股 {symbol}:")
                        print(f"  当前换手率: {last_day['turnover_rate']:.2f}%")
                        print(f"  五日平均换手率: {last_day['turnover_ma5']:.2f}%")
                        print(f"  当前成交量: {last_day['volume']/1e4:.2f}万手")
                        print(f"  五日平均成交量: {last_day['volume_ma5']/1e4:.2f}万手")
                        
            except KeyError as e:
                if self.debug:
                    print(f"股票 {symbol} 数据字段缺失: {str(e)}")
            except IndexError as e:
                if self.debug:
                    print(f"股票 {symbol} 在 {date.date()} 无有效数据")
            except Exception as e:
                if self.debug:
                    print(f"处理 {symbol} 时发生未知错误: {str(e)}")
        
        # 排序逻辑
        candidates.sort(key=lambda x: x['turnover'], reverse=True)
        
        # 调试信息
        if self.debug:
            print(f"找到 {len(candidates)} 只符合条件的候选股")
            if len(candidates) > 0:
                print("前3名候选股详情:")
                for i, c in enumerate(candidates[:3]):
                    print(f"{i+1}. {c['symbol']} 价格:{c['price']:.2f} 换手率:{c['turnover']:.2f}%")
        
        # 执行买入
        available_cash = min(self.current_cash * 0.1, 10000)
        bought_count = 0
        
        for stock in candidates[:3]:  # 最多买入前3名
            symbol = stock['symbol']
            price = stock['price']
            
            # 计算可买数量
            max_lots = available_cash // (price * 100 * 1.0003)
            if max_lots < 1:
                continue
                
            buy_shares = max_lots * 100
            cost = buy_shares * price * 1.0003
            
            if cost > available_cash:
                continue
                
            # 更新持仓
            new_position = pd.DataFrame([{
                'symbol': symbol,
                'shares': buy_shares,
                'buy_price': price,
                'buy_date': date
            }])
            self.positions = pd.concat([self.positions, new_position], ignore_index=True)
            
            # 更新资金
            self.current_cash -= cost
            available_cash -= cost
            bought_count += 1
            
            # 记录交易
            self.trade_log.append({
                'date': date,
                'symbol': symbol,
                'type': 'buy',
                'shares': buy_shares,
                'price': price
            })
            
            if self.debug:
                print(f"成功买入 {symbol} {buy_shares}股 单价:{price:.2f}")
        
        # 最终调试信息
        if self.debug:
            print(f"本日实际买入 {bought_count} 只股票")
            print(f"剩余可用资金: {self.current_cash:.2f}元")
            print("=== 买入处理完成 ===")

    def _record_daily_value(self, date):
        """记录每日净值"""
        date_str = date.date()
        position_value = 0
        if not self.positions.empty:
            prices = self.positions['symbol'].map(self.price_cache[date_str])
            position_value = (self.positions['shares'] * prices).sum()
            
        total_value = self.current_cash + position_value
        self.daily_values.append({'date': date, 'value': total_value})

    def generate_report(self):
        """生成优化后的报告"""
        df = pd.DataFrame(self.daily_values)
        df['return'] = df['value'].pct_change()
        
        total_return = (df.iloc[-1]['value'] / self.initial_capital - 1) * 100
        annualized = (df['value'].iloc[-1]/self.initial_capital)**(252/len(df)) - 1
        max_drawdown = (df['value'].cummax() - df['value']).max() / df['value'].cummax().max()
        
        print(f"\n{' 回测报告 ':=^40}")
        print(f"▪ 时间范围: {self.start_date.date()} - {self.end_date.date()}")
        print(f"▪ 初始资金: {self.initial_capital:,.2f}元")
        print(f"▪ 最终价值: {df.iloc[-1]['value']:,.2f}元")
        print(f"▪ 总收益率: {total_return:.2f}%")
        print(f"▪ 年化收益: {annualized*100:.2f}%")
        print(f"▪ 最大回撤: {max_drawdown*100:.2f}%")
        print(f"▪ 成交次数: {len(self.trade_log)}次")
        
        return df

if __name__ == "__main__":
    engine = OptimizedBacktestEngine(
        start_date="2023-01-01",
        end_date="2023-03-31",
        initial_capital=100000
    )
    engine.debug = True
    # 初始化引擎（数据预加载）
    print("正在初始化引擎...")
    engine.initialize_engine()
    
    # 运行回测
    print("开始回测...")
    engine.run_backtest()
    
    # 生成报告
    report = engine.generate_report()
    
    # 可视化
    report.set_index('date')['value'].plot(
        title='资金曲线',
        figsize=(12, 6),
        grid=True,
        rot=45
    )