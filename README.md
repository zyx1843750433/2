# Battery SOC Project (Android Daily-Use Dataset)

本项目实现了一个**连续时间**的 SOC（电池剩余电量）模型，并把 Android 手机日常使用数据里
的 **屏幕、CPU、网络、GPS、后台任务** 等因素通过一个可解释的方式作用到功率 `P(t)` 上。

> 你给的“锂电池模型.pdf”里提出：基础情况可把 `P(t)` 当常数；进一步可对 `P(t)` 进行线性加权改造，
> 并保持 `I(t)` 通过联立方程求得。本项目就按这个思路实现。

---

## 1. 模型（核心方程）

SOC 连续时间微分方程：

- `dSOC/dt = - I(t) / (C_eff * 3600)`

功率、电流、电压耦合：

- `I(t) = P(t) / V_term(t)`
- `V_term(t) = E_ocv(SOC) - I(t) * R_int(SOC, T)`

联立可得到二次方程：

- `R I^2 - E I + P = 0`（取放电物理解）

本项目用 `scipy.solve_ivp` 数值积分得到 `SOC(t)`。

---

## 2. 关键数据来自哪里（压缩包里的哪些文件）

压缩包 `A dataset from the daily use of features in Android devices.zip` 里有 4 类数据：

1) **Dynamic data**（最关键）：每秒（或近似每秒）一行的动态数据  
   包含 `battery_level / battery_current / battery_voltage / battery_power`，以及
   `screen_status / bright_level / cpu_usage / wifi_* / mobile_* / gps_* / ...`

2) **Background data**：`backgroundAPPS.csv`  
   记录后台 App 列表（用 `;` 分隔），我们把它转为 `bg_app_count`

3) **Static data**：`static.csv`  
   含 `battery_capacity (mAh)`，用于给 `C_nom` 提供更合理的容量基准

4) **Application data**：App 列表映射（本项目暂不做 App 分类建模，可后续扩展）

---

## 3. 数据清洗（无效数据处理）

- `timestamp` 混合格式：动态数据一般是 `YYYY-MM-DD ...`，后台数据常见 `DD-MM-YYYY ...`  
  项目使用 `robust_to_datetime()` 自动推断
- `battery_current` 单位不统一：有些文件是 **A**（如 -0.3），有些是 **mA**（如 -570）  
  项目会根据 95% 分位的量级自动识别并换算到 **A**
- `collected` 字段：有些天大量 `collected=0`（可能是缺失/插值/不可靠样本）  
  - 拟合功率权重默认只用 `collected==1`（更可信）
  - 但基线功率 `P0` 会从“看起来很闲”的样本中估计（screen off、cpu 很低、gps off、网络低）

---

## 4. 让额外因素影响功率：可解释线性模型

功率模型采用（可解释）线性加权：

`P(t) = P0 + w_screen*screen_on + w_bright*(screen_on*brightness) + w_cpu*cpu + w_net*net + w_wifiweak*wifi_weak + w_gps*gps + w_bg*bg`

- `P0`：常数基线功耗（idle/待机）
- 其余项：由数据拟合得到的权重（W），并约束为非负（物理上“多开功能不会让功耗变小”）

拟合方法：`Ridge(positive=True)`，带一点正则避免过拟合。

---

## 5. 如何运行（PyCharm / 命令行）

### 5.0 最省事的运行方式（不传参数）

1) 在项目根目录创建文件夹：`data/raw/`  
2) 把你下载的压缩包（不要解压）放进去，并**建议改名**为：`data/raw/android_dataset.zip`  
3) 直接运行（不带任何参数）：

```bash
python src/main.py
```

程序会自动：
- 在 `data/raw/` / `data/` / 项目根目录里寻找数据集 zip
- 自动挑选一个最完整的 `(date, device)`（优先同时有 background/static 的那组）
- 完成：清洗 → 拟合功率模型 → SOC 仿真 → 输出 `results/`

> 也可以用环境变量指定数据路径：  
> `ANDROID_DATASET_ZIP=...`，可选 `ANDROID_DATASET_DATE`、`ANDROID_DATASET_DEVICE`。


### 5.1 环境依赖

建议使用 conda / venv 安装：

```
pip install -r requirements.txt
```

### 5.2 列出 zip 里有哪些 (date, device)

```
python src/main.py list --zip "data/raw/A dataset from the daily use of features in Android devices.zip"
```

### 5.3 一键跑通（拟合功率 + SOC 仿真 + 画图）

示例（你可以先用 dt=10s 加速）：

```
python src/main.py run \
  --zip "data/raw/A dataset from the daily use of features in Android devices.zip" \
  --date 20230224 --device 70a09b5174d07fff \
  --dt 10
```

输出会保存在 `results/`：
- `power_model.json`：拟合出的 `P0` 和权重
- `sim_results.csv`：每个时间点的 measured vs simulated
- `power.png / soc.png / voltage.png`
- `metrics.json`：误差指标

---

## 6. 下一步可扩展（你后续可以做）

- 把 OCV 曲线 / 内阻模型参数对该设备进行数据拟合（更贴近真实手机电压曲线）
- 引入非线性：例如屏幕亮度可能是凸函数，网络弱信号的惩罚也可能非线性
- 使用更细粒度的后台任务特征：结合 Application data，把后台 App 分类（社交/视频/游戏）做不同权重
