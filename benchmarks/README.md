# Benchmarks

`leaderboard.json` 是进入 Git 的唯一冠军成绩登记表。原始逐样本分数和日志保存在本地 `artifacts/`，不提交到仓库。

更新冠军记录时必须同时更新模型文件、配置、方案说明和提交包哈希。`local_validation` 与 `online_leaderboard` 分开记录，禁止用本地分数冒充线上成绩。
