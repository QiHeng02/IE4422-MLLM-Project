from ie4422.boxes import greedy_match, iou, scale_box, smart_resize, yolo_to_xyxy
from ie4422.io import parse_answer

B = ["talking", "turning_head", "using_phone", "using_computer"]


def test_yolo_to_xyxy_and_clip():
    assert yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 100, 50) == [40.0, 15.0, 60.0, 35.0]
    assert yolo_to_xyxy(0.0, 0.0, 0.2, 0.2, 100, 100)[:2] == [0.0, 0.0]


def test_smart_resize_multiple_of_28_and_cap():
    h, w = smart_resize(1080, 1920, 3136, 1003520)
    assert h % 28 == 0 and w % 28 == 0 and h * w <= 1003520
    assert smart_resize(560, 840, 3136, 1003520) == (560, 840)  # already fits


def test_iou_and_matching():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert greedy_match([[0, 0, 10, 10], [0, 0, 10, 10]], [[0, 0, 10, 10]], 0.5) == 1  # one-to-one
    assert scale_box([10, 10, 20, 20], (100, 100), (200, 50)) == [20, 5, 40, 10]


def test_parse_answer_variants():
    ok = parse_answer('```json\n{"flagged": [{"bbox_2d": [1,2,3,4], "behaviour": "Using phone"}], "reason": "x"}\n```', B)
    assert ok["flagged"] == [{"behaviour": "using_phone", "bbox_2d": [1, 2, 3, 4]}]
    assert parse_answer('Sure! {"flagged": [], "reason": "none"}', B)["flagged"] == []
    assert parse_answer("no json here", B) is None
    assert parse_answer('{"flagged": "oops"}', B) is None
    dropped = parse_answer('{"flagged": [{"bbox_2d": [1,2,3,4], "behaviour": "sleeping"}]}', B)
    assert dropped["flagged"] == []
