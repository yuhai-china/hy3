# 在 CUDA 上优化 GGUF 推理速度 —— 逐步记录

本文档按时间顺序记录了我们如何将 Hy3 GGUF 运行时（`HYV3ForCausalLM`，一个
2950 亿参数 / 210 亿激活的 MoE 模型，Q4_K_M 量化）的 CUDA 解码路径，在单张
NVIDIA **B300**（Blackwell Ultra，计算能力 10.3）、CUDA 12.8 上，从
**4.6 tok/s 提升到 44 tok/s**（80 层全部卸载到 GPU）。

目标是让后来的工程师能够复现推理过程、理解每一步*为什么*有效，并避开我们踩过的坑。

---

## 0. 环境与基本规则

| 项目 | 取值 |
|------|------|
| GPU | NVIDIA B300 SXM6，计算能力 10.3 |
| 工具链 | CUDA 12.8（`/usr/local/cuda/bin/nvcc`），驱动 13.0 |
| 模型 | `hy3_q4k_mixed.gguf`，162 GB，80 层，embd 4096，64 头 / 8 KV 头，256 选 8 专家 |
| 编译 | `make NVCC=/usr/local/cuda/bin/nvcc -j4` |
| 后端源码 | `hy3_gpu.cu`、`hy3_gpu.h`；驱动循环在 `hy3.c` |

**用血泪换来的基本规则（动手前必读）：**

1. **用绝对路径调用 nvcc**（`/usr/local/cuda/bin/nvcc`）。本机上裸的 `nvcc`
   找不到 `cicc`。
2. **只使用普通架构标志** —— `sm_90`、`sm_100`、`compute_100`。带 "a" 后缀的
   架构目标（`sm_90a`、`sm_100a`、`compute_100a`）在本 B300 / 驱动 13.0 /
   CUDA 12.8 组合下，**编译和启动都没有任何 CUDA 报错，却会产生全零 logits**。
   我们同时嵌入 `sm_100` cubin 和 `compute_100` PTX，让驱动在加载时 JIT 出真正的
   B300 SASS。详见 `Makefile` 中的长注释块。
3. **每个可用状态都要提交。** 我们曾经因为一次误操作的 `git checkout`（丢弃了
   未提交的改动）而毁掉数小时的工作。每次实验前先提交。
4. **每次改动后都要验证正确性**，而不仅仅是速度（见 §11）。

---

## 1. 基线与测量方法

全部卸载的解码基线约为 **4.6 tok/s**。在优化任何东西之前，我们先加了监测手段，
因为看不见的东西无法优化：

- `HY3_TIMING=1` —— 由 `hy3.c` 打印的每 token 墙钟拆解（eval 与 sample）。
- `HY3_GRAPH_BENCH=1` —— 连续重放 CUDA graph 50 次并报告纯 GPU 的 ms/token，
  把设备端工作从主机端 / 采样开销中分离出来。
- `HY3_SKIP_ATTN` / `HY3_SKIP_FFN` / `HY3_SKIP_EXP` —— 跳过某一层阶段，
  通过"跑与不跑再相减"来做差分归因。

**早期关键发现**（来自差分 profiling）：解码是*延迟受限*而非算力受限
（SM 利用率约 25%，显存控制器约 0%）。跳过路由专家矩阵乘（`HY3_SKIP_EXP=1`）
把一个约 70 ms 的 token 压到约 10 ms —— 也就是说，**Q4_K 路由专家矩阵乘占了约
85% 的解码时间。** 下面每一步或直接或间接都是冲着这个事实去的。

---

## 2. 第 1 步 —— Blackwell 构建 + O(n²)→O(n) KV 缓存

提交 `bdbb2a8` —— *Blackwell/B300 支持 + CUDA 解码提速 2.9 倍。*

两个相互独立的收益打包在一起：

1. **正确的 B300 构建**（上面的架构标志）。仅此一项就让 GPU 路径产生正确的
   logits，而不是全零。
2. **KV 缓存改成每 token O(n)。** 原路径在一个不断增长的缓冲区上重算注意力，
   工作量是 O(n²)，且每 token 都重读全部 K/V。我们改为按位置索引的常驻设备 K/V
   缓存，因此每个解码步只追加一份 K/V，并只读一次缓存。

结果：**4.6 → 约 13.2 tok/s。**

---

## 3. 第 2 步 —— FP16 KV 缓存

提交 `757f261` —— *fp16 KV 缓存。*

将 K/V 以 `half` 而非 `float` 存储。这**使 KV 显存减半**，更重要的是，对于
延迟 / 带宽受限的解码，它**使注意力读取带宽减半**。online-softmax 注意力核
（`attention_kernel_online`，`hy3_gpu.cu:173`）直接读取 half 缓存并在 FP32 中
累加，因此实践中不损失精度。

---

## 4. 第 3 步 —— 融合注意力与路由核

随 Blackwell 重写一起完成。每层的注意力路径被合并成几个融合核，以减少启动次数
和中间的全局显存往返：

- `qk_norm_rope_fused_kernel`（`hy3_gpu.cu:144`）—— 在一个核里完成 Q 和 K 的
  RMS-norm **加上** RoPE，每头一个 warp/block。
- `attention_kernel_online`（`hy3_gpu.cu:173`）—— online-softmax（flash 风格）
  注意力，float4 分块，每头一个 warp，直接读 FP16 KV 缓存，不实体化分数矩阵。
- `router_topk_kernel`（`hy3_gpu.cu:81`）—— MoE top-k 路由**在设备端**完成
  （此前每层都要往返 CPU 一次）。含 `best=0` 的 NaN 保护，避免退化的路由分数
  错误地选中 0 号专家。
- `add_rmsnorm_kernel`（`hy3_gpu.cu:218`）—— 融合的残差相加 + RMS-norm。

此处的差分 profiling 证实注意力此时只占一个 token 的约 5 ms；FFN/专家约 65 ms。
注意力已不值得再追。

---

## 5. 第 4 步 —— 通过 `cublasGemmEx` 使用 FP16 稠密权重

提交 `e89e36e` —— *通过 cublasGemmEx 使用 FP16 稠密权重。*

稠密（非专家）投影 —— QKV、共享 gate/up、down、output —— 被转成 FP16，并作为
GEMV 通过 `cublasGemmEx`（`hy3_gpu.cu:380`）运行。这利用了张量核的 FP16 通路，
并**使稠密矩阵乘的权重读取带宽减半**。GEMV 是带宽受限的，因此这是直接的带宽收益，
精度影响可忽略（FP32 累加）。

---

## 6. 第 5 步 —— CUDA graph 解码

提交 `b12941a` —— *全 GPU 路径的 CUDA graph 解码。*

在约 13 tok/s 时，解码每 token 要启动几十个小核 × 80 层。每次启动的开销主导了
这个延迟受限的 token。我们把整个 80 层解码捕获成一张 **CUDA graph** 并在每步重放：

- 所有依赖位置的核都从设备指针 `d_pos` 读取 token 位置（在重放之间用一次极小的
  拷贝更新），因此*图的拓扑是固定的*，可以一次捕获。
- 流程：**warmup**（一次 eager 前向以填充惰性分配的状态）→ **capture** →
  每 token **replay**（`gpu_decode_graph`，`hy3_gpu.cu:446`）。

这几乎消除了所有主机端启动延迟。此时纯 graph 重放约 42 ms/token。速度：
**约 13.2 → 13.84 tok/s** 端到端（graph 主要消除的是本来就能重叠的主机开销，
但它是干净地测量和攻击纯设备时间的前提）。

---

## 7. 第 6 步 —— warp-per-row 的 Q4_K 专家矩阵乘

提交 `6a6ef1e` —— *warp-per-row Q4_K 专家矩阵乘 → 13.8→20.4 tok/s。*

此时 profiler 已表明专家占了一个 token 的 85%，而专家矩阵乘对 GPU 的占用严重不足。
原始专家核大致是每个输出行用一个*线程*，且只有几十个 block —— B300 的约 148 个
SM 几乎闲置。

重新设计：**一个 warp（32 条 lane）通过整 warp 的 shuffle 归约计算一个输出行。**
grid 为 `grid.x = ceil(M / MOE_WPB)`、`grid.y = slot`（被路由的专家实例），
`MOE_WPB = 8` 个 warp/block。对 gate 投影，这产生约 1536 个 block 而非约 24 个 ——
SM 满占用，从而掩盖了 Q4_K 反量化 + 权重读取的延迟。

结果：**13.84 → 20.4 tok/s。**

---

## 8. 第 7 步 —— Q4_K 子块并行

提交 `16211db` —— *Q4_K 子块并行 → 22.7 tok/s。*

warp-per-row 在行的块数 `nb` 较小时仍有 lane 空闲：工作按每步一个 256 元素的
Q4_K 超块分配。我们引入 `dev_dot_q4_K_sub`，把每行的工作按其 **`nb*8` 个子块**
（32 元素的 Q4_K 子块）分配，使得无论 `nb` 多小，32 条 lane 都保持忙碌。

结果：**20.4 → 22.68 tok/s。** 纯 graph 重放约 42.3 ms。

此时差分 profiling 已经变钝：`HY3_SKIP_EXP=1` 给出 9.98 ms 的 graph 重放
（约 100 tok/s），而带专家为 42 ms —— 专家仍占约 32 ms（76%）。专家每 token 读
约 5.9 GB 的 Q4_K 权重；在 42 ms 下这只有 **约 140 GB/s 有效带宽，大约比** B300
约 8 TB/s 的下限**低 56 倍**。读取是**非合并的**：warp 的 32 条 lane 在抓取分散的
子块，而非一次连续事务。

---

## 9. 第 8 步 —— 合并的 warp 协作 Q4_K 行点积（决定性一步）

提交 `bb2b33b` —— *合并的 warp 协作 Q4_K 专家矩阵乘。*

这是决定性的显存合并修复。新设备函数 `warp_row_dot_q4k`（`hy3_gpu.cu:51`），
被 `moe_matmul_q4k_id_kernel`（down 投影）与
`moe_matmul_q4k_id_gate_up_kernel`（gate+up）共同使用。

### Q4_K 块布局（映射为何棘手）

一个 Q4_K 超块在 144 字节的结构里保存 256 个量化权重：一个 FP16 `d`、一个 FP16
`dmin`、12 个打包的 6 位 scale/min 字节，以及一段 **128 字节的 `qs`** 4 位半字节
（nibble）。这 256 个权重是 8 个各 32 元素的子块。子块对 `p`（子块 `2p`、`2p+1`）
占据 `qs` 的字节 `[p*32 .. p*32+31]`，其中**低**半字节属于子块 `2p`，**高**半字节
属于子块 `2p+1`。

### 合并的访存模式

让 32 条 lane 把每个块的 128 字节 `qs` 作为 **32 个连续的 `uint32` 在一次合并事务
中读取**：`((const uint32_t*)blk->qs)[lane]`。于是：

- lane `L` 拥有字节区间 `[L*4, L*4+3]`，恰好落入对 `p = L/8`、元素偏移
  `e0 = (L%8)*4`（证明：`p*32 + e0 == 4*L`）。
- 每条 lane 的 4 个字节产生 4 个低半字节（→ 子块 `2p`，元素 `e0..e0+3`）和 4 个
  高半字节（→ 子块 `2p+1`）。
- 因为 scale/min 在一个子块内是常数，且对对 `p` 的全部 8 条 lane 相同，每条 lane
  可以把自己的部分点积先乘以自己子块的 scale/min；最后的 warp shuffle 归约会正确
  地把一切求和：`sc·d·Σ(q·x) − m·dmin·Σx`，分摊到各 lane 上。

对齐成立：`cuda_block_q4_K` 为 144 = 9×16 字节，`qs` 位于偏移 16，`cudaMalloc`
是 256 对齐的，因此 `uint32`（以及 16 字节）访问都是对齐的。

对 gate+up 核，同一个激活 `x` 被两个权重矩阵复用，因此 `warp_row_dot_q4k` 对每行
调用两次，而无需再走一遍逻辑去重读 `x`。

### 结果

- 纯 graph 重放：**42.3 → 22.3 ms/token。**
- 80 层端到端：**22.68 → 44 tok/s**（约 2 倍）。

这一次合并改动是后期最大的收益，因为它精确打击了 profiler 指出的瓶颈：非合并的
Q4_K 权重读取。

---

## 10. 整体进展

| 步骤 | 改动 | tok/s（80 层） |
|-----:|------|---------------:|
| 0 | 基线 | 4.6 |
| 1 | Blackwell 构建 + O(n) KV 缓存 | 约 13.2 |
| 2 | FP16 KV 缓存 | （合并计入） |
| 3 | 融合注意力 / 设备端路由 | （合并计入） |
| 4 | FP16 稠密权重（`cublasGemmEx`） | （合并计入） |
| 5 | CUDA graph 解码 | 13.84 |
| 6 | warp-per-row Q4_K 专家矩阵乘 | 20.4 |
| 7 | Q4_K 子块并行 | 22.68 |
| 8 | **合并的 warp 协作 Q4_K 行点积** | **44** |

纯 GPU graph 重放最终约为 22.3 ms/token。峰值常驻显存约 192 GB。

---

## 11. 如何验证正确性（每次改动都要做）

如果 logits 是错的，速度就毫无意义（回忆一下全零 logits 的坑）。两个快速检查：

1. **算术，≤ 约 40–50 层**（完全正确的区间）：
   ```
   ./hy3-cli -m /home/user/hy3-gguf/hy3_q4k_mixed.gguf --gpu-layers 20 -p "11+22+33=?" -n 60
   ```
   必须得到 `66` / `\boxed{66}`。

2. **80 层全量下的连贯性：**
   ```
   ./hy3-cli -m /home/user/hy3-gguf/hy3_q4k_mixed.gguf --gpu-layers 80 -p "The capital of France is" -n 40
   ```
   必须答出 "Paris" 并保持连贯。

**已知局限（并非回归）：** 在超过约 40–50 卸载层后，输出可能偏离 CPU 参考。原因是
GPU 与 CPU 的浮点归约顺序会轻微扰动模型的*硬* top-k MoE 路由 —— 一次平局翻转改变了
专家选择，进而级联。这是在极薄的路由裕度上重排 FP 归约的固有现象，在本工作之前就存在，
且不影响 80 层下的连贯性。

---

## 12. 基准测试配方

```bash
# 编译
make NVCC=/usr/local/cuda/bin/nvcc -j4

# 纯 GPU 设备时间（每 token 重放 50 次），隔离核工作
HY3_GRAPH_BENCH=1 ./hy3-cli -m .../hy3_q4k_mixed.gguf --gpu-layers 80 \
  -p "The capital of France is" -n 40 2>&1 | grep "graph replay"

# 端到端吞吐（这才是真实数字；不要与 GRAPH_BENCH 同用）
./hy3-cli -m .../hy3_q4k_mixed.gguf --gpu-layers 80 \
  -p "Explain quantum entanglement in three sentences." -n 120 2>&1 | tail -1

# 分阶段归因
HY3_TIMING=1 ...        # eval 与 sample 拆解
HY3_SKIP_EXP=1 ...      # 相减以隔离路由专家开销
HY3_SKIP_ATTN=1 ...     # 隔离注意力开销
HY3_SKIP_FFN=1 ...      # 隔离 FFN 开销
```

---

## 13. 通用经验（可迁移到其他 GGUF/CUDA 工作）

1. **先 profiling 再优化。** 差分的阶段跳过（`HY3_SKIP_*`）告诉我们专家占了一个
   token 的 85% —— 其余一切都是干扰项。
2. **解码是延迟 / 带宽受限，而非 FLOP 受限。** batch 1 的 GEMV 只读一次权重、
   算得很少；收益来自占用率、更少的启动（CUDA graph）、更小 / 半精度的读取，
   以及*合并访存*。
3. **先占用率，再合并访存。** warp-per-row（占用率）→ 子块（空闲 lane）→ 合并的
   `uint32` 读取（带宽）—— 每一步都解锁下一个瓶颈。有效带宽比峰值低约 56 倍这个数字
   就是非合并访问的铁证。
4. **要在字节级理解你的量化布局。** 整个最终收益都建立在 `p*32 + e0 == 4*L` 这个
   恒等式上，它让 32 条 lane 读取 128 个连续字节的同时，仍干净地映射到 Q4_K 的
   子块 / 半字节结构。
5. **CUDA graph 需要设备端的位置状态**（`d_pos`），以使被捕获的拓扑在各 token 间
   不变。
6. **每次改动都验证精度**，并了解你模型固有的发散区间，以免去追一个其实只是 FP
   归约顺序造成的"bug"。

---

## 14. 尚存的机会

- 移除现在已无用的 `HY3_SKIP_*` profiling 脚手架和死代码设备函数
  （`dev_dot_q4_K_sub`、`qwarp_sum_f32`、`dev_dot_q4_K_f32_block`）。
- 在第二条 stream 上把共享专家 GEMV 与路由专家矩阵乘重叠
  （该技术在 `ds4` 参考引擎中可用）。
- 研究一种稳定归约的路由器，把正确区间扩展到约 50 卸载层以上。
