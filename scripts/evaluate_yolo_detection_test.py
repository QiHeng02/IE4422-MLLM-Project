import json,hashlib
from pathlib import Path
from ultralytics import YOLO
import ultralytics
export=json.loads(Path('outputs/yolo/final_test/detection_export.json').read_text())
weights=Path('runs/detect/outputs/yolo/v8s/weights/best.pt')
assert hashlib.sha256(weights.read_bytes()).hexdigest()==export['checkpoint_sha256']
model=YOLO(str(weights))
r=model.val(data='data/processed/yolo_detection_test/data.yaml',split='test',imgsz=640,batch=16,workers=4,device=0,conf=0.001,iou=0.7,max_det=300,plots=True,project=str(Path('outputs/yolo/detection_metrics').resolve()),name='test',exist_ok=False)
result={'metric':'raw behaviour detection before fixed rules','split':'test','images':export['images'],'checkpoint_sha256':export['checkpoint_sha256'],'ultralytics':ultralytics.__version__,'conf_floor_for_PR_curve':0.001,'nms_iou':0.7,'imgsz':640,'max_det':300,'mAP50':float(r.box.map50),'mAP50_95':float(r.box.map),'by_class':{model.names[int(cid)]:{'AP50':float(r.box.ap50[i]),'AP50_95':float(r.box.ap[i])} for i,cid in enumerate(r.box.ap_class_index)},'framework_metrics':{k:float(v) for k,v in r.results_dict.items()},'save_dir':str(r.save_dir),'notes':'Includes phone boxes in detection AP; differs from flagging box metrics which exclude using_phone. Partially annotated SCB behaviours can count unannotated predictions as false positives. No test threshold/model selection.'}
Path('results/metrics/yolo_detection__test.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
