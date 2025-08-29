# tools/collect_distributions.py (mmyolo/mmengine 스타일 예시, 네 런처에 맞게 조정)
import torch
from mmengine.config import Config
from mmdet.apis import init_detector
from mmengine.runner import Runner

cfg = Config.fromfile("configs/open_world/sowod/our/large_t1.py")

# 파이프라인의 att_select가 select_att()를 호출하지 않게 잠시 막거나, log_start_epoch=999로 미룸
for i,p in enumerate(cfg.get("pipline", [])):
    if p.get("type") == "att_select":
        cfg.pipline[i]["log_start_epoch"] = 0  # 로깅만
        cfg.pipline[i]["__collect_only__"] = True  # 훅에서 select 호출 안 하도록 너 코드에 가드 추가해도 됨

runner = Runner.from_cfg(cfg)
runner.train()  # 1 epoch 설정 권장

# after one epoch, 헤드에서 distributions 꺼내 저장 (헤드에 enable_log 호출 필요시 호출)
# (만약 자동으로 저장되게 구현되어 있으면 이 파트 생략)
head = runner.model.module.bbox_head if hasattr(runner.model, "module") else runner.model.bbox_head
torch.save({
  "positive_distributions": head.positive_distributions,
  "negative_distributions": head.negative_distributions
}, "paper_repeat/sowod/previous_log/sowod_distribution_sim1.pth")
print("saved distributions.")
