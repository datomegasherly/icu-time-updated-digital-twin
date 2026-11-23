import csv
import gzip
import json
import re
from datetime import datetime
from pathlib import Path


# داده‌های کامل روی سیستم شما؛ این پوشه‌ها جابه‌جا نمی‌شوند.
EICU_DIR = Path(r"F:\Articles\1\data\eICU")
MIMIC_HOSP_DIR = Path(r"F:\Articles\1\data\MIMIC\hosp")
MIMIC_ICU_DIR = Path(r"F:\Articles\1\data\MIMIC\icu")

# خود اسکریپت و خروجی آن در پوشهٔ project قرار می‌گیرند.
PROJECT_DIR = Path(r"F:\Articles\1\project")
OUTPUT_FILE = PROJECT_DIR / "audit_icu.json"

MIMIC_HOSP_TABLES = (
    "patients",
    "admissions",
    "labevents",
    "d_labitems",
)

MIMIC_ICU_TABLES = (
    "icustays",
    "chartevents",
    "d_items",
    "inputevents",
    "procedureevents",
)

EICU_TABLES = (
    "patient",
    "lab",
    "vitalPeriodic",
    "vitalAperiodic",
    "nurseCharting",
    "infusionDrug",
    "treatment",
)


def find_table(folder: Path, table_name: str):
    """یک جدول CSV یا CSV.GZ را در پوشه یا زیرپوشه‌های آن پیدا می‌کند."""
    valid_names = {
        f"{table_name.lower()}.csv",
        f"{table_name.lower()}.csv.gz",
    }

    matches = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.name.lower() in valid_names
    ]

    if len(matches) > 1:
        locations = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"چند نسخه از جدول {table_name} پیدا شد:\n{locations}\n"
            "برای جلوگیری از مخلوط شدن نسخه‌های داده، فقط یک نسخه را "
            "در مسیر تعیین‌شده نگه دار یا مسیر پوشه را دقیق‌تر کن."
        )

    return matches[0] if matches else None


def open_csv(path: Path):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


def get_table_info(path):
    """فقط ساختار جدول را می‌خواند؛ داده‌های ردیفی در گزارش ذخیره نمی‌شوند."""
    if path is None:
        return {"present": False}

    with open_csv(path) as stream:
        columns = next(csv.reader(stream), [])

    return {
        "present": True,
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "columns": columns,
    }


def iter_rows(path: Path, required_columns):
    with open_csv(path) as stream:
        reader = csv.DictReader(stream)
        available = set(reader.fieldnames or [])
        missing = set(required_columns) - available

        if missing:
            raise ValueError(
                f"ستون‌های ضروری در {path.name} وجود ندارند: "
                f"{sorted(missing)}"
            )

        yield from reader


def parse_iso_datetime(value: str):
    return datetime.fromisoformat((value or "").strip())


def parse_eicu_adult_age(value):
    """
    True:  بزرگسال
    False: کمتر از ۱۸ سال
    None:  سن نامشخص یا غیرقابل‌تفسیر
    """
    text = (value or "").strip()

    if re.fullmatch(r"\d+", text):
        return int(text) >= 18

    # مانند > 89؛ حد پایین این بازه قطعاً بزرگسال است.
    match = re.fullmatch(r">\s*(\d+)", text)
    if match:
        lower_bound = int(match.group(1))
        return True if lower_bound >= 17 else None

    return None


def audit_mimic(patients_path, icustays_path):
    if patients_path is None or icustays_path is None:
        raise FileNotFoundError(
            "برای MIMIC، فایل patients.csv(.gz) در پوشهٔ hosp "
            "و فایل icustays.csv(.gz) در پوشهٔ icu ضروری است."
        )

    # فقط هنگام پردازش در حافظه نگه داشته می‌شود؛ در خروجی نوشته نمی‌شود.
    patient_age_anchors = {}
    patient_rows = 0
    patients_with_valid_age_anchor = 0

    for row in iter_rows(
        patients_path,
        ("subject_id", "anchor_age", "anchor_year"),
    ):
        patient_rows += 1

        try:
            subject_id = row["subject_id"].strip()
            anchor_age = int(row["anchor_age"].strip())
            anchor_year = int(row["anchor_year"].strip())

            if not subject_id or anchor_age < 0:
                continue

            patient_age_anchors[subject_id] = (anchor_age, anchor_year)
            patients_with_valid_age_anchor += 1
        except (ValueError, TypeError, AttributeError):
            continue

    summary = {
        "patient_rows": patient_rows,
        "patients_with_valid_age_anchor": patients_with_valid_age_anchor,
        "icu_stays_all": 0,
        "icu_stays_with_estimable_age_at_icu": 0,
        "icu_stays_age_18_plus": 0,
        "icu_stays_age_18_plus_los_24h_plus": 0,
        "icu_stays_with_missing_or_invalid_age_or_intime": 0,
        "adult_stays_with_missing_or_invalid_outtime": 0,
        "adult_stays_with_nonpositive_duration": 0,
    }

    for row in iter_rows(
        icustays_path,
        ("subject_id", "intime", "outtime"),
    ):
        summary["icu_stays_all"] += 1

        try:
            subject_id = row["subject_id"].strip()
            anchor_age, anchor_year = patient_age_anchors[subject_id]
            intime = parse_iso_datetime(row["intime"])

            # سن تقریبی هنگام ورود به ICU بر اساس سالِ تاریخ جابه‌جاشده.
            age_at_icu = anchor_age + (intime.year - anchor_year)
        except (KeyError, ValueError, TypeError, AttributeError):
            summary["icu_stays_with_missing_or_invalid_age_or_intime"] += 1
            continue

        summary["icu_stays_with_estimable_age_at_icu"] += 1

        if age_at_icu < 18:
            continue

        summary["icu_stays_age_18_plus"] += 1

        try:
            outtime = parse_iso_datetime(row["outtime"])
            duration_hours = (outtime - intime).total_seconds() / 3600
        except (ValueError, TypeError, AttributeError):
            summary["adult_stays_with_missing_or_invalid_outtime"] += 1
            continue

        if duration_hours <= 0:
            summary["adult_stays_with_nonpositive_duration"] += 1
        elif duration_hours >= 24:
            summary["icu_stays_age_18_plus_los_24h_plus"] += 1

    return summary


def audit_eicu(patient_path):
    if patient_path is None:
        raise FileNotFoundError(
            "فایل patient.csv یا patient.csv.gz در مسیر eICU پیدا نشد."
        )

    hospital_ids = set()
    summary = {
        "icu_stays_all": 0,
        "icu_stays_age_18_plus": 0,
        "icu_stays_age_18_plus_los_24h_plus": 0,
        "icu_stays_with_unknown_age": 0,
        "adult_stays_with_missing_or_invalid_los": 0,
        "adult_stays_with_nonpositive_duration": 0,
        "hospitals_with_recorded_id": 0,
    }

    for row in iter_rows(
        patient_path,
        ("patientunitstayid", "age", "unitdischargeoffset", "hospitalid"),
    ):
        summary["icu_stays_all"] += 1

        hospital_id = (row["hospitalid"] or "").strip()
        if hospital_id:
            hospital_ids.add(hospital_id)

        adult = parse_eicu_adult_age(row["age"])

        if adult is None:
            summary["icu_stays_with_unknown_age"] += 1
            continue
        if not adult:
            continue

        summary["icu_stays_age_18_plus"] += 1

        try:
            # eICU: زمان ترخیص از ICU نسبت به پذیرش، بر حسب دقیقه.
            duration_minutes = float(row["unitdischargeoffset"])
        except (ValueError, TypeError):
            summary["adult_stays_with_missing_or_invalid_los"] += 1
            continue

        if duration_minutes <= 0:
            summary["adult_stays_with_nonpositive_duration"] += 1
        elif duration_minutes >= 24 * 60:
            summary["icu_stays_age_18_plus_los_24h_plus"] += 1

    summary["hospitals_with_recorded_id"] = len(hospital_ids)
    return summary


def main():
    for folder in (EICU_DIR, MIMIC_HOSP_DIR, MIMIC_ICU_DIR):
        if not folder.is_dir():
            raise FileNotFoundError(f"پوشهٔ داده پیدا نشد: {folder}")

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    mimic_files = {
        name: find_table(MIMIC_HOSP_DIR, name)
        for name in MIMIC_HOSP_TABLES
    }
    mimic_files.update({
        name: find_table(MIMIC_ICU_DIR, name)
        for name in MIMIC_ICU_TABLES
    })

    eicu_files = {
        name: find_table(EICU_DIR, name)
        for name in EICU_TABLES
    }

    report = {
        "mimic": {
            "tables": {
                name: get_table_info(path)
                for name, path in mimic_files.items()
            },
            "summary": audit_mimic(
                mimic_files["patients"],
                mimic_files["icustays"],
            ),
        },
        "eicu": {
            "tables": {
                name: get_table_info(path)
                for name, path in eicu_files.items()
            },
            "summary": audit_eicu(eicu_files["patient"]),
        },
        "method_note": (
            "این فقط ممیزی اولیه است. شمارش اقامت‌های حداقل ۲۴ ساعت "
            "توصیفی است و هنوز معیار ورود نهایی پژوهش نیست. "
            "سن MIMIC هنگام ورود به ICU با دقت سالانه برآورد شده است. "
            "هیچ مدل پیش‌بینی در این مرحله آموزش داده نشده است."
        ),
    }

    OUTPUT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nممیزی اولیه با موفقیت تمام شد.")
    print(f"فایل خروجی: {OUTPUT_FILE}")

    for database in ("mimic", "eicu"):
        print(f"\n{database.upper()} SUMMARY:")
        print(
            json.dumps(
                report[database]["summary"],
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
