from bottle_detector.models import normalize_detection_payload


def test_normalize_detection_payload_fills_arabic_labels() -> None:
    payload = {
        "defects": [
            {
                "type": "dirt",
                "description_ar": "",
            }
        ],
        "summary_ar": "",
    }

    normalized = normalize_detection_payload(payload)

    assert normalized.defects[0].type == "dirty"
    assert normalized.defects[0].label_ar == "العلبة متسخة بالطين او التراب"
    assert normalized.defects[0].description_ar
    assert normalized.summary_ar
    assert normalized.confidence == 0.8


def test_normalize_detection_payload_drops_cap_defects() -> None:
    payload = {
        "defects": [{"type": "cap_defect", "description_ar": "no cap"}],
        "summary_ar": "",
        "confidence": 1.4,
    }

    normalized = normalize_detection_payload(payload)

    assert normalized.defects == []
    assert normalized.confidence == 1.0


def test_normalize_detection_payload_recognizes_factory_defect() -> None:
    payload = {
        "defects": [{"type": "molding", "description_ar": ""}],
        "summary_ar": "",
    }

    normalized = normalize_detection_payload(payload)

    assert normalized.defects[0].type == "factory_defect"
    assert normalized.defects[0].label_ar == "عيب تصنيعي واضح وكبير في شكل العلبة"
