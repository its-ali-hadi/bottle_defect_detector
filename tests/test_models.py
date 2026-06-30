from bottle_detector.models import normalize_detection_payload


def test_normalize_detection_payload_fills_arabic_labels() -> None:
    payload = {
        "defects": [
            {
                "type": "cap",
                "description_ar": "",
            }
        ],
        "summary_ar": "",
    }

    normalized = normalize_detection_payload(payload)

    assert normalized.defects[0].type == "cap_defect"
    assert normalized.defects[0].label_ar == "غطاء علبة بيه مشكلة"
    assert normalized.defects[0].description_ar
    assert normalized.summary_ar
