import akshare as ak

# 查找与黄金相关的函数
print("查找与黄金相关的函数：")
gold_funcs = [func for func in dir(ak) if 'gold' in func.lower()]
for func in gold_funcs:
    print(f"  {func}")

# 查找与基金相关的函数
print("\n查找与基金相关的函数：")
fund_funcs = [func for func in dir(ak) if 'fund' in func.lower()]
for func in fund_funcs[:20]:  # 只显示前20个
    print(f"  {func}")

# 查找与汇率相关的函数
print("\n查找与汇率相关的函数：")
exchange_funcs = [func for func in dir(ak) if 'exchange' in func.lower() or 'currency' in func.lower()]
for func in exchange_funcs:
    print(f"  {func}")