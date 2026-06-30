from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


DEFAULT_INPUT = Path("outputs/detections.json")
DEFAULT_OUTPUT_DIR = Path("result")
LATEST_OUTPUT_NAME = "detections_latest.xlsx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert bottle detector JSON results to an Excel workbook.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input detections JSON path. Default: outputs/detections.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Default: result",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional exact Excel file name. If omitted, creates a timestamped file and detections_latest.xlsx.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = export_excel_report(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        output_name=args.output_name,
    )
    print(f"Wrote Excel report: {result['report_path']}")
    if result.get("latest_path"):
        print(f"Updated latest Excel report: {result['latest_path']}")
    return 0


def export_excel_report(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_name: str | None = None,
) -> dict[str, Path | None]:
    data = load_json(input_path)
    workbook = build_workbook(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / (output_name or timestamped_output_name())
    workbook.save(report_path)

    latest_path: Path | None = None
    if output_name is None:
        latest_path = output_dir / LATEST_OUTPUT_NAME
        try:
            shutil.copyfile(report_path, latest_path)
        except PermissionError:
            latest_path = None

    return {"report_path": report_path, "latest_path": latest_path}


def timestamped_output_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"detections_{stamp}.xlsx"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must contain an object at the top level.")
    if not isinstance(payload.get("detections"), list):
        raise ValueError('Input JSON must contain a "detections" list.')
    return payload


def build_workbook(data: dict[str, Any]) -> Workbook:
    workbook = Workbook()
    report_sheet = workbook.active
    report_sheet.title = "Report"

    detections = data.get("detections", [])
    add_full_report_sheet(report_sheet, data, detections)
    summary_sheet = workbook.create_sheet("Summary")
    add_summary_sheet(summary_sheet, data, detections)
    add_bottle_details_sheet(workbook, detections)

    for sheet in workbook.worksheets:
        sheet.sheet_view.rightToLeft = True
        apply_common_styles(sheet)
    return workbook


def add_full_report_sheet(sheet: Any, data: dict[str, Any], detections: list[dict[str, Any]]) -> None:
    defect_counts = Counter(
        defect.get("type", "unknown")
        for item in detections
        for defect in item.get("defects", [])
    )
    summary_rows = [
        ("المصدر", data.get("source", "")),
        ("المودل", data.get("model", "")),
        ("تاريخ إنشاء JSON", data.get("created_at", "")),
        ("تاريخ تصدير Excel", datetime.now().isoformat(timespec="seconds")),
        ("عدد العلب الكلي", len(detections)),
        ("عيوب جسم العلبة", defect_counts.get("body_defect", 0)),
        ("عيوب الغطاء", defect_counts.get("cap_defect", 0)),
        ("علب متسخة", defect_counts.get("dirty", 0)),
    ]

    sheet.append(["الإحصائية", "القيمة"])
    for row in summary_rows:
        sheet.append(list(row))
    add_table(sheet, "ReportSummaryTable", start_row=1, end_row=sheet.max_row, end_column=2)

    sheet.append([])
    details_headers = [
        "تسلسل العلبة",
        "عدد العيوب",
        "العيوب بالتفصيل",
        "ملخص العلبة",
    ]
    sheet.append(details_headers)
    details_header_row = sheet.max_row
    for item in detections:
        defects = item.get("defects", [])
        sheet.append(
            [
                item.get("sequence", ""),
                len(defects),
                format_defects_for_bottle(defects),
                item.get("summary_ar", ""),
            ]
        )
    add_table(
        sheet,
        "ReportBottleDetailsTable",
        start_row=details_header_row,
        end_row=sheet.max_row,
        end_column=len(details_headers),
    )


def add_summary_sheet(sheet: Any, data: dict[str, Any], detections: list[dict[str, Any]]) -> None:
    defect_counts = Counter(
        defect.get("type", "unknown")
        for item in detections
        for defect in item.get("defects", [])
    )
    rows = [
        ("Source", data.get("source", "")),
        ("Model", data.get("model", "")),
        ("JSON Created At", data.get("created_at", "")),
        ("Excel Exported At", datetime.now().isoformat(timespec="seconds")),
        ("Total Bottles", len(detections)),
        ("Body Defects", defect_counts.get("body_defect", 0)),
        ("Cap Defects", defect_counts.get("cap_defect", 0)),
        ("Dirty Bottles", defect_counts.get("dirty", 0)),
    ]
    sheet.append(["Field", "Value"])
    for row in rows:
        sheet.append(list(row))
    add_table(sheet, "SummaryTable")


def add_bottle_details_sheet(workbook: Workbook, detections: list[dict[str, Any]]) -> None:
    sheet = workbook.create_sheet("Bottle Details")
    headers = [
        "تسلسل العلبة",
        "عدد العيوب",
        "العيوب بالتفصيل",
        "ملخص العلبة",
    ]
    sheet.append(headers)
    for item in detections:
        defects = item.get("defects", [])
        sheet.append(
            [
                item.get("sequence", ""),
                len(defects),
                format_defects_for_bottle(defects),
                item.get("summary_ar", ""),
            ]
        )
    add_table(sheet, "BottleDetailsTable")


def add_table(
    sheet: Any,
    name: str,
    *,
    start_row: int = 1,
    end_row: int | None = None,
    end_column: int | None = None,
) -> None:
    end_row = end_row or sheet.max_row
    end_column = end_column or sheet.max_column
    end_column_letter = get_column_letter(end_column)
    ref = f"A{start_row}:{end_column_letter}{end_row}"
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    sheet.add_table(table)


def apply_common_styles(sheet: Any) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E2F3")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    header_rows = find_header_rows(sheet)
    for row_number in header_rows:
        for cell in sheet[row_number]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in sheet.iter_rows():
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    sheet.freeze_panes = "A2" if sheet.title != "Report" else "A11"
    auto_size_columns(sheet)


def find_header_rows(sheet: Any) -> set[int]:
    header_markers = {
        "Field",
        "Sequence",
        "الإحصائية",
        "تسلسل العلبة",
    }
    rows = {1}
    for row in sheet.iter_rows():
        first_value = row[0].value
        if first_value in header_markers:
            rows.add(row[0].row)
    return rows


def auto_size_columns(sheet: Any) -> None:
    for column_cells in sheet.columns:
        column_letter = get_column_letter(column_cells[0].column)
        max_length = 0
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_length = max(max_length, len(value))
        sheet.column_dimensions[column_letter].width = min(max(max_length + 3, 12), 60)


def format_defects_for_bottle(defects: list[dict[str, Any]]) -> str:
    if not defects:
        return "لا توجد عيوب"
    lines: list[str] = []
    for index, defect in enumerate(defects, start=1):
        label = defect.get("label_ar") or defect_type_ar(defect.get("type", ""))
        defect_type = defect.get("type", "")
        description = defect.get("description_ar", "")
        lines.append(
            f"{index}. العيب: {label}"
            f"\n   النوع: {defect_type}"
            f"\n   الوصف: {description}"
        )
    return "\n".join(lines)


def defect_type_ar(defect_type: Any) -> str:
    values = {
        "body_defect": "زرف او عيب في العلبة",
        "cap_defect": "غطاء علبة بيه مشكلة",
        "dirty": "العلبة متسخة",
    }
    return values.get(str(defect_type), str(defect_type))


if __name__ == "__main__":
    raise SystemExit(main())
