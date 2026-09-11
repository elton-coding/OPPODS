# V269 启动前换行校验恢复

de3a30a已预登记推送。首次入口在验证V268历史绑定时exit1，尚未创建V269计划、探针目录或正式目录，无训练步骤执行。主干切换导致13个已跟踪文本由LF变CRLF。对每个差异文件只做CRLF→LF候选转换，先证明转换后的SHA256严格等于旧冻结计划，再恢复字节；没有豁免哈希，没有修改算法或旧计划。

恢复后再次验证历史绑定，入口仍使用同一预登记代码；这是训练开始前环境恢复，不是正式重跑或续训。原失败证据：RuntimeError('frozen inputs changed; do not continue this experiment')，发生run_shared_lr_v269.py的verify_bound_inputs(previous)；当时PLAN/OUTPUT/PROBE三者均不存在。
