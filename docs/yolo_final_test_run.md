# YOLO Final Test 执行记录

用户于 2026-10-21 明确要求开始 Final Test 并记录步骤。只使用已在验证集选好的配置，不根据 Test 分数调参。

1. 检查 `outputs/yolo/validation_selection.json`：选择为 conf=0.1，checkpoint 为 `runs/detect/outputs/yolo/v8s/weights/best.pt`。不重新训练、不搜索 Test 阈值。
2. 检查此前不存在 `yolo_rules_test*.jsonl` 预测或对应测试指标，避免覆盖已有结果。
3. 核验输入：`test.jsonl` 为 755 张图、2,265 个问题；`test_reworded.jsonl` 为相同 755 张图、4,530 个问题。各文件 qid 唯一，图片存在，split 均为 test。测试图片路径与 train_single、val 无交集。这是路径级检查，不是新执行的近似图片检查。
4. 保存 checkpoint、问题文件、规则表、配置、场景标签、推理/评分/解析等代码及作业脚本的 SHA-256 到 `outputs/yolo/final_test/manifest.json`。记录 Git commit；仓库已有未提交的实验记录和验证结果未被清理。
5. 创建 `jobs/yolo_test.sbatch`：单张 PRO 6000，最长 1 小时，使用现有 `.venv`。`set -x` 将实际执行命令写入 Slurm 日志；记录 GPU、驱动、主机、UTC 时间和完整 pip 环境。
6. 在执行推理前，`outputs/yolo/final_test/check.py before` 验证所有冻结输入哈希，并拒绝覆盖已存在的预测文件。
7. 对两个问题文件依次运行原 `scripts/yolo_rules_baseline.py`，固定 conf=0.1、imgsz=640。YOLO 检测后按原 `configs/rules.csv` 和问题行为列表过滤，输出 `results/predictions/yolo_rules_test.jsonl` 与 `yolo_rules_test_reworded.jsonl`。
8. 每份预测完成后，用未改动的 `scripts/evaluate.py --method yolo_rules` 评分，写入 `results/metrics/yolo_rules__test.json`、`yolo_rules__test_reworded.json`，并追加 `results/summary.csv`。
9. 最后执行 `check.py after`：再次验证输入哈希；检查预测 qid 完整、唯一且与问题精确对应；检查 JSON 可解析、阈值固定；比较同一图片/情境的标准及改写版输出是否一致。结果及预测/指标文件哈希保存为 `outputs/yolo/final_test/completion.json`。
10. 提交命令：`sbatch jobs/yolo_test.sbatch`。作业 ID：217766。日志：`logs/yolo_test-217766.out`。提交不代表完成；以 Slurm 状态、日志和 completion.json 为准。

## 核查入口

- 完整命令：`jobs/yolo_test.sbatch`
- 输入版本和哈希：`outputs/yolo/final_test/manifest.json`
- 实际执行日志：`logs/yolo_test-217766.out`
- 运行依赖：`outputs/yolo/final_test/environment.txt`（作业开始后生成）
- 完成后的自动检查：`outputs/yolo/final_test/completion.json`
- 正式指标：`results/metrics/yolo_rules__test.json` 与 `yolo_rules__test_reworded.json`

## 比较边界

这次测试使用本地重建并锁定的测试集合。与 QiHeng 实际运行的 Qwen 测试问题清单是否完全一致，尚无原始文件可直接核验。规则后主指标为问题×行为的 Flagging 指标；框定位指标另算，using_phone 按现有配置不参与框指标。Participation 规则列由人预先提供，不能把其表现解释为 YOLO 的文本理解能力。

Final Test completion confirmed: Slurm 217766 COMPLETED, exit 0:0, elapsed 4m26s. Both standard (2,265 questions) and reworded (4,530 questions) outputs passed input hash, unique/exact qid and JSON checks. Standard/reworded predictions match per image/context. known_precision=0.8533, known_recall=0.6738, known_f1=0.7530; all-context flag_f1=0.8684; participation unseen_f1=0.9336. Participation uses a human-provided rule column; these results are not evidence of text understanding. No test-driven tuning performed. See completion.json and results/metrics/yolo_rules__test*.json.

### 2026-10-21 · YOLO raw detection test mAP · Slurm 217824

User requested test mAP50. Exported the same 755 test images into separate `data/processed/yolo_detection_test/` with original manifest boxes for all five behaviours, using original export_yolo.py conversion helpers. Counts: talking 181, hand_raising 1568, turning_head 741, using_phone 29, reading_notes 24. No fixed-rule filtering or GT flagging applied.
Command: `sbatch jobs/yolo_detection_test.sbatch`; runner `outputs/yolo/final_test/detection_eval.py`. Same locked best.pt, imgsz 640, batch16, conf floor 0.001 for AP curve, NMS IoU 0.7, max_det300; split=test. This confidence floor is not a replacement for val-selected flagging threshold0.1. Original training/flagging files unchanged. Input hashes and mapping: `outputs/yolo/final_test/detection_export.json`. Output: `results/metrics/yolo_detection__test.json`; plots `outputs/yolo/detection_metrics/test/`. Includes phone boxes; partial SCB annotation limitation applies. No model or threshold tuning. Status: submitted, not yet confirmed complete.

Raw detection mAP GPU change: user requested another GPU. Previous pending PRO 6000 job 217824 cancelled without running; replacement A6000 job 217857 submitted via `sbatch jobs/yolo_detection_test.sbatch`. Same checkpoint, test images, labels and mAP parameters; no retraining or threshold changes. Account eligibility and scheduler estimates checked before replacement. Log: `logs/yolo_map_test-217857.out`.

Raw detection mAP completion confirmed: A6000 job 217857 COMPLETED, exit0:0, elapsed3m12s. Same 755 held-out images and locked checkpoint. Test mAP50=0.3227949714; mAP50-95=0.1898673731. AP50: talking0.4839207507, turning_head0.1717379827, hand_raising0.5903606838, using_phone0.0452459083, reading_notes0.3227095313. Source: `results/metrics/yolo_detection__test.json`; plots: `outputs/yolo/detection_metrics/test/`; log: `logs/yolo_map_test-217857.out`. These are raw detection metrics before fixed rules, distinct from flagging known_f1=0.7530.

## GitHub source packaging

Before publishing, the executed helper sources were copied without logic changes into `scripts/check_yolo_final_test.py` and `scripts/evaluate_yolo_detection_test.py`; committed job scripts now reference these source paths. Historical execution logs and frozen manifests refer to the original helpers under outputs/. Frozen historical hashes must not be rewritten to match the packaging change. Both helpers require local generated artifacts under `outputs/yolo/final_test/`; datasets, exports, manifests, predictions, checkpoints and environment remain local and are not committed. The original one-off test label export was performed as described above using export_yolo.py helpers; its image mapping and input hashes remain in detection_export.json. Scripts are not standalone dataset download/preparation tools.
